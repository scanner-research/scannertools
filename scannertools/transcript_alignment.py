from .prelude import *
from scannerpy import FrameType, Kernel
from typing import Sequence

import numpy as np
import pickle
import pysrt
import codecs
import math
import sys
import os
import cv2
import logging
import time
import re
import tempfile
import scipy.io.wavfile as wavf
import multiprocessing
import json
import traceback
import gentle

MISSING_THRESH = 0.2

"""
Help functions for fid, time, second transfer
"""
def fid2second(fid, fps):
    second = 1. * fid / fps
    return second

def time2second(time):
    if len(time) == 3:
        return time[0]*3600 + time[1]*60 + time[2]
    elif len(time) == 4:
        return time[0]*3600 + time[1]*60 + time[2] + time[3] / 1000.0

def second2time(second, sep=','):
    h, m, s, ms = int(second) // 3600, int(second % 3600) // 60, int(second) % 60, int((second - int(second)) * 1000)
    return '{:02d}:{:02d}:{:02d}{:s}{:03d}'.format(h, m, s, sep, ms)


class TranscriptAligner():
    def __init__(self, seg_length=60, max_misalign=10, num_thread=8, exhausted=False, 
                 transcript_path=None, media_path=None, align_dir=None):
        self.seg_length = seg_length
        self.text_shift = max_misalign
        self.num_thread = num_thread
        self.exhausted = exhausted
        self.transcript_path = transcript_path
        self.media_path = media_path
        self.align_dir = align_dir
        self.sequential = True if self.media_path is None else False
        
        self.audio_shift = 1
        self.seg_idx = 0
        self.punctuation_all = ['>>', ',', ':', '[.]', '[?]']
        self.num_words = 0
        
        if not self.media_path is None:
            _, ext = os.path.splitext(self.media_path)
            self.video_name = os.path.basename(self.media_path)
            if ext == '.mp4':
                cap = cv2.VideoCapture(self.media_path)
                self.video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.video_length = int(self.video_frames // self.fps)
                self.num_seg = int(self.video_length // self.seg_length)
            elif ext == '.wav':
                raise Exception("Not implemented error")
    
    def load_transcript(self, transcript_path):
        """"
        Load transcript from *.srt file
        """
        # Check file exist
        if not os.path.exists(transcript_path):
#             transcript_path = transcript_path.replace('cc5', 'cc1') 
#             if not os.path.exists(transcript_path):
            raise Exception("Transcript file does not exist")

        # Check encoded in uft-8
        try:
            file = codecs.open(transcript_path, encoding='utf-8', errors='strict')
            for line in file:
                pass
        except UnicodeDecodeError:
            raise Exception("Transcript not encoded in utf-8")

        transcript = []
        subs = pysrt.open(transcript_path)
        text_length = 0
        num_words = 0
        for sub in subs:
            transcript.append((sub.text, time2second(tuple(sub.start)[:3]), time2second(tuple(sub.end)[:3])))
            text_length += transcript[-1][2] - transcript[-1][1]
            for w in sub.text.replace('.', ' ').replace('-', ' ').split():
                num_words += 1 if w.islower() or w.isupper() else 0
        print('Num of words in transcript:',  num_words)

#         Check transcript completeness     
#             if 1. * text_length / video_desp['video_length'] < MIN_TRANSCRIPT:
#                 raise Exception("Transcript not complete")
        self.transcript = transcript
        self.num_words = num_words

    def extract_transcript_segment(self, seg_idx, large_shift=0):
        start = seg_idx * self.seg_length - self.text_shift + large_shift
        end = (seg_idx + 1) * self.seg_length + self.text_shift + large_shift
        punctuation_seg = []
        transcript_seg = ''
        for (text, ss, ee) in self.transcript:
            if ss >= end:
                break
            if ss >= start:
                offset = len(transcript_seg)
                for p in self.punctuation_all:
                    pp = p.replace('[', '').replace(']', '')
                    for match in re.finditer(p, text):
                        punctuation_seg.append((match.start()+offset, pp))
                transcript_seg += text + ' ' 
        punctuation_seg.sort()
        return transcript_seg, punctuation_seg
    
    def extract_transcript_all(self):
        self.text_seg_list = []
        self.punc_seg_list = []
        for seg_idx in range(self.num_seg):
            transcript_seg, punctuation_seg = self.extract_transcript_segment(seg_idx)
            self.text_seg_list.append(transcript_seg)
            self.punc_seg_list.append(punctuation_seg)
    
    def extract_audio_segment(self, seg_idx): 
        start = seg_idx * self.seg_length
        start = start - self.audio_shift if seg_idx > 0 else start
        duration = self.seg_length 
        duration += self.audio_shift * 2 if seg_idx > 0 else self.audio_shift
        cmd = 'ffmpeg -i ' + self.media_path + ' -vn -acodec copy '
        cmd += '-ss {:d} -t {:d} '.format(start, duration)
        audio_path = tempfile.NamedTemporaryFile(suffix='.aac').name
        cmd += audio_path
        os.system(cmd)
        return audio_path
    
    def extract_audio_all(self):
        pool = multiprocessing.Pool(self.num_thread)
        self.audio_seg_list = pool.map(self.extract_audio_segment, [i for i in range(self.num_seg)])
    
    def align_segment_thread(self, seg_idx):
        def exhausted_align_segment(seg_idx):
            # first do alignment with -+ self.text_shift    
            transcript, punctuation = self.extract_transcript_segment(seg_idx, 0)
            result_seg = self.align_segment(seg_idx, self.audio_seg_list[seg_idx], transcript, punctuation)    
            if len(result_seg['align_word_list']) < 10 or  \
                1 - 1. * result_seg['num_word_aligned'] / len(result_seg['align_word_list']) < MISSING_THRESH:
                return result_seg
            
            best_result_seg = result_seg
            for start, end in [((seg_idx-1)*self.seg_length, (seg_idx+1)*self.seg_length), 
                               (seg_idx*self.seg_length, (seg_idx+2)*self.seg_length)]:   
                transcript, _ = self.extract_transcript(start, end)
                result_seg = self.align_segment(seg_idx, self.audio_seg_list[seg_idx], transcript, [])
                if result_seg['num_word_aligned'] > best_result_seg['num_word_aligned']:
                    best_result_seg = result_seg
            return best_result_seg
        
        if self.exhausted:
            return exhausted_align_segment(seg_idx)
        else:
            return self.align_segment(seg_idx, self.audio_seg_list[seg_idx], self.text_seg_list[seg_idx], self.punc_seg_list[seg_idx])
    
    def align_segment(self, seg_idx, audio_path, transcript, punctuation):
        args = {'log': 'INFO',
            'nthreads': 1 if not self.sequential else self.num_thread,
            'conservative': True,
            'disfluency': True,
            }
        disfluencies = set(['uh', 'um'])
#         with open(self.text_seg_list[seg_idx]) as fh:
#             transcript = fh.read()

        resources = gentle.Resources()
        with gentle.resampled(audio_path) as wavfile:
            aligner = gentle.ForcedAligner(resources, transcript, nthreads=args['nthreads'], disfluency=args['disfluency'], conservative=args['conservative'], disfluencies=disfluencies)
            result = aligner.transcribe(wavfile)
            aligned_seg = [word.as_dict(without="phones") for word in result.words]

        # insert punctuation
        start_idx = 0
        for offset, p in punctuation:
            for word_idx, word in enumerate(aligned_seg[start_idx:]):
                if word['case'] != 'not-found-in-transcript': 
                    if p == '>>' and (offset == word['startOffset'] - 3 or offset == word['startOffset'] - 4): 
                        word['word'] = '>> ' + word['word']
                        start_idx += word_idx
                        break
                    if p != '>>' and offset == word['endOffset']:
                        word['word'] = word['word'] + p
                        start_idx += word_idx
                        break
        
        # post-process
        align_word_list = []
        seg_start = seg_idx * self.seg_length
        seg_start = seg_start - self.audio_shift if seg_idx > 0 else seg_start
        seg_shift = self.audio_shift if seg_idx > 0 else 0

        enter_alignment = False
        word_missing = []
        num_word_aligned = 0
        for word_idx, word in enumerate(aligned_seg):
            if word['case'] == 'not-found-in-transcript':
                # align_word_list.append(('[Unknown]', (word['start'] + seg_start, word['end'] + seg_start)))
                pass
            elif word['case'] == 'not-found-in-audio':
                if enter_alignment:
                    word_missing.append(word['word'])
            else:
                assert(word['case'] == 'success')
                if word['start'] > self.seg_length + seg_shift:
                    break
                elif word['start'] >= seg_shift:
                    num_word_aligned += 1
                    enter_alignment = True
                    if len(word_missing) <= 2:
                        num_word_aligned += len(word_missing)
                    if len(word_missing) > 0:
                        start = align_word_list[-1][1][1]
                        end = word['start'] + seg_start
                        step = (end - start) / len(word_missing)
                        for i, w in enumerate(word_missing):
                            align_word_list.append(('{'+w+'}', (start+i*step, start+(i+1)*step)))
                        word_missing = []

                    align_word_list.append((word['word'], (word['start'] + seg_start, word['end'] + seg_start)))
        return {'align_word_list': align_word_list, 'num_word_aligned': num_word_aligned}    
        
    def run_all(self):
        self.load_transcript(self.transcript_path)
        self.extract_transcript_all()
        print("Extracting transcripts done")
        self.extract_audio_all()
        print("Extracting audio done")
        pool = multiprocessing.Pool(self.num_thread)
        self.result_all = pool.map(self.align_segment_thread, [i for i in range(self.num_seg)])
    
        align_word_list = []
        num_word_aligned = 0
        for seg_idx, seg in enumerate(self.result_all):
            align_word_list += [word for word in seg['align_word_list']]
            num_word_aligned += seg['num_word_aligned']
            print(seg_idx, seg['num_word_aligned'], len(seg['align_word_list']))
#             print(self.text_seg_list[seg_idx])
        print('word_missing: ', 1 - 1. * num_word_aligned / self.num_words)
        if not self.align_dir is None:
            output_path = os.path.join(self.align_dir, self.video_name + '.word.srt')
            TranscriptAligner.dump_aligned_transcript_byword(align_word_list, output_path)
            output_path = os.path.join(self.align_dir, self.video_name + '.align.srt')
            TranscriptAligner.dump_aligned_transcript(align_word_list, output_path)
        return {'word_missing': 1 - 1. * num_word_aligned / self.num_words}    
   
    def extract_transcript(self, start, end):
        transcript_seg = ''
        offset2time = {}
        for (text, ss, ee) in self.transcript:
            if ss >= end:
                break
            if ss >= start:
                offset = len(transcript_seg)
                words = text.replace('.', ' ').replace('-', ' ').split(' ')
                step = (ee - ss) / len(words)
                for i, w in enumerate(words):
                    offset2time[offset] = ss+i*step
                    offset += len(w) + 1
                transcript_seg += text + ' ' 
        return transcript_seg, offset2time    
    
    def extract_audio(self, start, end): 
        duration = end - start
        cmd = 'ffmpeg -i ' + self.media_path + ' -vn -acodec copy '
        cmd += '-ss {:d} -t {:d} '.format(start, duration)
        audio_path = tempfile.NamedTemporaryFile(suffix='.aac').name
        cmd += audio_path
        os.system(cmd)
        return audio_path
    
    def align_segment_simple(self, audio_path, audio_start, transcript, offset2time):
        args = {'log': 'INFO',
            'nthreads': 1 if not self.sequential else self.num_thread,
            'conservative': True,
            'disfluency': True,
            }
        disfluencies = set(['uh', 'um'])

        resources = gentle.Resources()
        with gentle.resampled(audio_path) as wavfile:
            aligner = gentle.ForcedAligner(resources, transcript, nthreads=args['nthreads'], disfluency=args['disfluency'], conservative=args['conservative'], disfluencies=disfluencies)
            result = aligner.transcribe(wavfile)
            aligned_seg = [word.as_dict(without="phones") for word in result.words]
        
        align_word_list = []
        num_word_aligned = 0
        for word_idx, word in enumerate(aligned_seg):
#             if word['case'] == 'not-found-in-transcript':
#                 pass
            if word['case'] == 'not-found-in-audio':
                align_word_list.append(('{'+word['word']+'}', audio_start, word['startOffset'])) 
#                 pass
            elif word['case'] == 'success':
                align_word_list.append((word['word'], word['start'] + audio_start, word['startOffset']))
                num_word_aligned += 1
                                       
        if len(align_word_list) == 0:
            return None
        for i, word in enumerate(align_word_list[len(align_word_list)//2 : ]):
            if word[2] in offset2time:
                estimate_shift = offset2time[word[2]] - word[1]
                break
#         print(num_word_aligned, estimate_shift)
#         print(align_word_list)
        return estimate_shift
    
    def search_text_shift(self, seg_idx):
        transcript, offset2time = self.extract_transcript((seg_idx-1)*self.seg_length, (seg_idx+2)*self.seg_length)
        
        estimate_shift = []
        for audio_shift in range(0, self.seg_length, self.text_shift):  
            audio_start = seg_idx * self.seg_length + audio_shift
            audio_path = self.extract_audio(audio_start, audio_start + self.text_shift)
            estimate = self.align_segment_simple(audio_path, audio_start, transcript, offset2time)
            if not estimate is None:
                estimate_shift.append(estimate)
        if len(estimate_shift) == 0:
            return None
        else:
            estimate_shift.sort()
            return estimate_shift[len(estimate_shift) // 2]
    
    def search_text_shift_all(self):
        pool = multiprocessing.Pool(self.num_thread)
        estimate_shift_list = pool.map(self.search_text_shift, [i for i in range(self.num_seg)])
        return estimate_shift_list

    def run_segment_scanner(self, audio, caption):
        def extract_transcript(seg_idx, caption, start, end):
            punctuation_seg = []
            transcript_seg = ''
            if seg_idx == 0:
                captions = caption[1] + caption[2]
            elif len(caption[1]) > 0 and len(caption[2]) > 0 and caption[1][0]['start'] == caption[2][0]['start']:
                # last segment
                captions = caption[0] + caption[1]
            else:
                captions = caption[0] + caption[1] + caption[2]
            for cap in captions:
                text, ss, ee = cap['line'], cap['start'], cap['end']
                if ss >= end:
                    break
                if ss >= start:
                    offset = len(transcript_seg)
                    for p in self.punctuation_all:
                        pp = p.replace('[', '').replace(']', '')
                        for match in re.finditer(p, text):
                            punctuation_seg.append((match.start()+offset, pp))
                    transcript_seg += text + ' ' 
            punctuation_seg.sort()
            return transcript_seg, punctuation_seg
        
        def dump_audio(seg_idx, audio):
            ar = audio[1].shape[0] // self.seg_length
            audio_shift = ar * self.audio_shift
            if seg_idx == 0:
                audio_cat = np.concatenate((audio[1], audio[2][:audio_shift, ...]), 0)
            else:
                audio_cat = np.concatenate((audio[0][-audio_shift:, ...], audio[1], audio[2][:audio_shift, ...]), 0)
            audio_path = tempfile.NamedTemporaryFile(suffix='.wav').name
            wavf.write(audio_path, ar, audio_cat)
            return audio_path    
        
        def exhausted_align_segment(seg_idx, audio_path, caption):
            # first do alignment with -+ self.text_shift    
            transcript, punctuation = extract_transcript(seg_idx, caption, 
                                                         seg_idx*self.seg_length - self.text_shift, 
                                                         (seg_idx+1)*self.seg_length + self.text_shift)
            result_seg = self.align_segment(seg_idx, audio_path, transcript, punctuation)    
            if len(result_seg['align_word_list']) < 10 or  \
                1 - 1. * result_seg['num_word_aligned'] / len(result_seg['align_word_list']) < MISSING_THRESH:
                return result_seg
            print(seg_idx)
            
            best_result_seg = result_seg
            for start, end in [((seg_idx-1)*self.seg_length, (seg_idx+1)*self.seg_length), 
                               (seg_idx*self.seg_length, (seg_idx+2)*self.seg_length)]:   
                transcript, punctuation = extract_transcript(seg_idx, caption, start, end)
                result_seg = self.align_segment(seg_idx, audio_path, transcript, punctuation)
                if result_seg['num_word_aligned'] > best_result_seg['num_word_aligned']:
                    best_result_seg = result_seg
            return best_result_seg
        
        if len(caption[1]) == 0:
            return {'align_word_list': [], 'num_word_aligned': 0, 'num_word_total': 0}
        else:
            seg_idx = int(caption[1][0]['start'] // self.seg_length)
        audio_path = dump_audio(seg_idx, audio)
        if self.exhausted:
            result_seg = exhausted_align_segment(seg_idx, audio_path, caption)
        else:
            transcript, punctuation = extract_transcript(seg_idx, caption, 
                                                         seg_idx*self.seg_length - self.text_shift, 
                                                         (seg_idx+1)*self.seg_length + self.text_shift)
            result_seg = self.align_segment(seg_idx, audio_path, transcript, punctuation)
        # count total number of words
        num_words = 0
        for line in caption[1]:
            for w in line['line'].replace('.', ' ').replace('-', ' ').split():
                num_words += 1 if w.islower() or w.isupper() else 0
        result_seg['num_word_total'] = num_words
        return result_seg
    
    
    @staticmethod
    def dump_aligned_transcript(align_word_list, path):
        SRT_INTERVAL = 1
        outfile = open(path, 'w')
        start, end = None, None
        srt_idx = 1
        for idx, word in enumerate(align_word_list):
            if start is None:
                start, end = word[1]
                text = word[0] + ' '
                continue
            if word[1][0] > start + SRT_INTERVAL:
                line = str(srt_idx) + '\n'
                line += '{:s} --> {:s}\n'.format(second2time(start), second2time(end))
                line += text + '\n\n'
                outfile.write(line)
                start, end = word[1]
                text = word[0] + ' '
                srt_idx += 1
            else:
                text += word[0] + ' '
                end = word[1][1]
        line = str(srt_idx) + '\n'
        line += '{:s} --> {:s}\n'.format(second2time(start), second2time(end))
        line += text + '\n\n'
        outfile.write(line)
        outfile.close()
    
    @staticmethod
    def dump_aligned_transcript_byword(align_word_list, path):
        outfile = open(path, 'w')
        srt_idx = 1
        for idx, word in enumerate(align_word_list):
            start, end = word[1]
            line = str(srt_idx) + '\n'
            line += '{:s} --> {:s}\n'.format(second2time(start), second2time(end))
            line += word[0] + '\n\n'
            outfile.write(line)
            srt_idx += 1
        outfile.close()  
        

@scannerpy.register_python_op(name='AlignTranscript', stencil=[-1,0,1])
class AlignTranscript(Kernel):
    def __init__(self, config):
        seg_length = config.args.get('seg_length', 60)
        text_shift = config.args.get('max_misalign', 10)
        num_thread = config.args.get('num_thread', 8)
        exhausted = config.args.get('exhausted', False)
        self.aligner = TranscriptAligner(seg_length, text_shift, num_thread, exhausted)
    
    def new_stream(self, args):
#         print(args['video_name'])
        pass
        
    def execute(self, audio: Sequence[FrameType], captions: Sequence[bytes]) -> bytes:
        cap_json = [json.loads(cap.decode('utf-8')) for cap in captions]
        result_seg = self.aligner.run_segment_scanner(audio, cap_json)
        return pickle.dumps(result_seg)


def parse(buf, config):
#     if len(buf) < 20 or len(buf) > 15000:
#         print('++++ len of buffer', len(buf))
    try: 
        return pickle.loads(buf)
    except Exception as e:
        traceback.print_exc()
        return {'align_word_list': [], 'num_word_aligned': 0, 'num_word_total': 0}
    
class AlignTranscriptPipeline(Pipeline):
    job_suffix = 'align_transcript'
    base_sources = ['audio', 'captions']
    run_opts = {'pipeline_instances_per_node': 8, 'io_packet_size': 4, 'work_packet_size': 4, 'checkpoint_frequency': 20}
    custom_opts = ['seg_length', 'max_misalign', 'num_thread', 'exhausted']
#     parser_fn = lambda _: lambda buf, _: pickle.loads(buf)
    parser_fn = lambda _: parse

    def build_pipeline(self):
        return {
            'align_transcript':
            self._db.ops.AlignTranscript(
                audio=self._sources['audio'].op,
                captions=self._sources['captions'].op,
                seg_length=self._custom_opts['seg_length'],
                max_misalign=self._custom_opts['max_misalign'],
                num_thread=self._custom_opts['num_thread'],
                exhausted=self._custom_opts['exhausted']
                )
        }

    def _build_jobs(self, cache):
        jobs = super(AlignTranscriptPipeline, self)._build_jobs(cache)
#         for (job, video_name) in zip(jobs, self._custom_opts['video_name']):
#             job._op_args[self._output_ops['align_transcript']] = {'video_name': video_name}
        return jobs

    def build_sink(self):
        return BoundOp(
            op=self._db.sinks.Column(columns=self._output_ops),
            args=[
                '{}_{}'.format(arg['path'], self.job_suffix)
                for arg in self._sources['audio'].args
            ])

align_transcript_pipeline = AlignTranscriptPipeline.make_runner()

def align_transcript(db, video_list, audio, caption, cache=False, align_dir=None, res_path=None):
    result = align_transcript_pipeline(db=db, audio=audio, captions=caption, cache=cache)
    if align_dir is None or res_path is None:
        return 

    if not res_path is None and os.path.exists(res_path):
        res_stats = pickle.load(open(res_path, 'rb'))
    else:
        res_stats = {}
    for idx, res_video in enumerate(result):
        video_name = video_list[idx]
        align_word_list = []
        num_word_aligned = 0
        num_word_total = 0
        if res_video is None:
            continue
        res_video_list = res_video.load()
        for seg_idx, seg in enumerate(res_video_list):
            align_word_list += [word for word in seg['align_word_list']]
            num_word_aligned += seg['num_word_aligned']
            if 'num_word_total' in seg:
                num_word_total += seg['num_word_total']
            else:
                num_word_total += len(seg['align_word_list'])
#             print(seg_idx, seg['num_word_aligned'])
        res_stats[video_name] = {'word_missing': 1 - 1. * num_word_aligned / num_word_total}
        print('word_missing', 1 - 1. * num_word_aligned / num_word_total)
#         print('word_missing', 1 - 1. * num_word_aligned / len(align_word_list))
        if not align_dir is None:
            output_path = os.path.join(align_dir, video_name + '.word.srt')
            TranscriptAligner.dump_aligned_transcript_byword(align_word_list, output_path)
            output_path = os.path.join(align_dir, video_name + '.align.srt')
            TranscriptAligner.dump_aligned_transcript(align_word_list, output_path)
        if not res_path is None:
            pickle.dump(res_stats, open(res_path, 'wb'))
