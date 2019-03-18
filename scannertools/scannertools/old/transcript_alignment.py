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


#----------Help functions for fid, time, second transfer----------
def fid2second(fid, fps):
    second = 1. * fid / fps
    return second


def time2second(time):
    return time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 1000.0


def second2time(second, sep=','):
    h, m, s, ms = int(second) // 3600, int(second % 3600) // 60, int(second) % 60, int(
        (second - int(second)) * 1000)
    return '{:02d}:{:02d}:{:02d}{:s}{:03d}'.format(h, m, s, sep, ms)


#---------Forced transcript-audio alignment using gentle----------
class TranscriptAligner():
    def __init__(self,
                 win_size=300,
                 seg_length=60,
                 max_misalign=10,
                 num_thread=1,
                 estimate=False,
                 transcript_path=None,
                 media_path=None,
                 align_dir=None):
        """
        @win_size: chunk size for estimating maximum mis-alignment
        @seg_length: chunk size for performing gentle alignment
        @max_misalign: maximum mis-alignment applied at the two ends of seg_length
        @num_thread: number of threads used in gentle
        @estimate: if True, run maximum mis-alignment estimate on each chunk of win_size
        @transcript_path: path to original transcript
        @media_path: path to video/audio
        @align_dir: path to save aligned transcript
        """
        self.win_size = win_size
        self.seg_length = seg_length
        self.text_shift = max_misalign
        self.num_thread = num_thread
        self.estimate = estimate
        self.transcript_path = transcript_path
        self.media_path = media_path
        self.align_dir = align_dir

        self.audio_shift = 1
        self.clip_length = 15
        self.seg_idx = 0
        self.punctuation_all = ['>>', ',', ':', '[.]', '[?]']
        self.num_words = 0

        if not self.media_path is None:
            _, ext = os.path.splitext(self.media_path)
            self.video_name = os.path.basename(self.media_path).split('.')[0]
            if ext == '.mp4':
                cap = cv2.VideoCapture(self.media_path)
                self.video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.video_length = int(self.video_frames // self.fps)
                self.num_seg = int(self.video_length // self.seg_length)
                self.num_window = int(self.video_length // self.win_size)
            elif ext == '.wav':
                raise Exception("Not implemented error")

    def load_transcript(self, transcript_path):
        """
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
            transcript.append((sub.text, time2second(tuple(sub.start)[:4]),
                               time2second(tuple(sub.end)[:4])))
            text_length += transcript[-1][2] - transcript[-1][1]
            for w in sub.text.replace('.', ' ').replace('-', ' ').split():
                num_words += 1 if w.islower() or w.isupper() else 0
        print('Num of words in transcript:', num_words)

        #         Check transcript completeness
        #             if 1. * text_length / video_desp['video_length'] < MIN_TRANSCRIPT:
        #                 raise Exception("Transcript not complete")
        self.transcript = transcript
        self.num_words = num_words

    def extract_transcript(self, start, end, offset_to_time=False):
        """
        extract transcript between [start, end) into a string, with additional offset to timestamp/puncuation
        """
        text_total = ''
        if offset_to_time:
            offset2time = {}
        else:
            offset2punc = []
        for (text, ss, ee) in self.transcript:
            if ss >= end:
                break
            if ss >= start:
                offset = len(text_total)
                if offset_to_time:
                    words = text.replace('.', ' ').replace('-', ' ').split(' ')
                    step = (ee - ss) / len(words)
                    for i, w in enumerate(words):
                        offset2time[offset] = ss + i * step
                        offset += len(w) + 1
                else:
                    for p in self.punctuation_all:
                        pp = p.replace('[', '').replace(']', '')
                        for match in re.finditer(p, text):
                            offset2punc.append((match.start() + offset, pp))
                text_total += text + ' '
        if offset_to_time:
            return text_total, offset2time
        else:
            offset2punc.sort()
            return text_total, offset2punc

    def extract_transcript_segment(self, seg_idx, large_shift=0):
        """
        extract transcript given specific segment
        """
        start = seg_idx * self.seg_length - self.text_shift - large_shift
        end = (seg_idx + 1) * self.seg_length + self.text_shift - large_shift
        return self.extract_transcript(start, end, offset_to_time=False)

    def extract_transcript_all(self):
        """
        extract transcript from all segments
        """
        self.text_seg_list = []
        self.punc_seg_list = []
        for seg_idx in range(self.num_seg):
            shift = self.shift_seg_list[seg_idx] if self.estimate else 0
            transcript_seg, punctuation_seg = self.extract_transcript_segment(seg_idx, shift)
            self.text_seg_list.append(transcript_seg)
            self.punc_seg_list.append(punctuation_seg)

    def extract_audio(self, start, end):
        """
        extract audio between [start, end] into a local .aac file
        """
        duration = end - start
        cmd = 'ffmpeg -i ' + self.media_path + ' -vn -acodec copy '
        cmd += '-ss {:d} -t {:d} '.format(start, duration)
        audio_path = tempfile.NamedTemporaryFile(suffix='.aac').name
        cmd += audio_path
        os.system(cmd)
        return audio_path

    def extract_audio_segment(self, seg_idx):
        """
        extract audio given specific segment
        """
        start = seg_idx * self.seg_length
        start = start - self.audio_shift if seg_idx > 0 else start
        duration = self.seg_length
        duration += self.audio_shift * 2 if seg_idx > 0 else self.audio_shift
        return self.extract_audio(start, start + duration)

    def extract_audio_all(self):
        """
        extract audio from all segments parallely
        """
        pool = multiprocessing.Pool(self.num_thread)
        self.audio_seg_list = pool.map(self.extract_audio_segment, [i for i in range(self.num_seg)])

    def gentle_solve(self, audio_path, transcript):
        """
        gentle wrapper to solve the forced alignment given audio file and text string 
        """
        args = {
            'log': 'INFO',
            'nthreads': self.num_thread,
            'conservative': True,
            'disfluency': True,
        }
        disfluencies = set(['uh', 'um'])
        resources = gentle.Resources()
        with gentle.resampled(audio_path) as wavfile:
            aligner = gentle.ForcedAligner(
                resources,
                transcript,
                nthreads=args['nthreads'],
                disfluency=args['disfluency'],
                conservative=args['conservative'],
                disfluencies=disfluencies)
            result = aligner.transcribe(wavfile)
        return [word.as_dict(without="phones") for word in result.words]

    def align_segment_thread(self, seg_idx):
        """
        function wrapped for multiprocessing
        """
        return self.align_segment(seg_idx, self.audio_seg_list[seg_idx],
                                  self.text_seg_list[seg_idx], self.punc_seg_list[seg_idx])

    def align_segment(self, seg_idx, audio_path, transcript, punctuation):
        """
        call gentle and post-process aligned results
        """
        aligned_seg = self.gentle_solve(audio_path, transcript)

        # insert punctuation
        start_idx = 0
        for offset, p in punctuation:
            for word_idx, word in enumerate(aligned_seg[start_idx:]):
                if word['case'] != 'not-found-in-transcript':
                    if p == '>>' and (offset == word['startOffset'] - 3
                                      or offset == word['startOffset'] - 4):
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
                assert (word['case'] == 'success')
                if word['start'] > self.seg_length + seg_shift:
                    break
                elif word['start'] >= seg_shift:
                    num_word_aligned += 1
                    enter_alignment = True
                    cur_start = word['start'] + seg_start
                    cur_end = word['end'] + seg_start
                    # make sure the prev_end <= cur_start
                    if len(align_word_list) > 0:
                        prev_end = align_word_list[-1][1][1]
                        if prev_end > cur_start and prev_end < cur_end:
                            cur_start = prev_end
                    # mis-aligned word handling
                    if len(word_missing) <= 2:
                        num_word_aligned += len(word_missing)
                    if len(word_missing) > 0:
                        step = (cur_start - prev_end) / len(word_missing)
                        for i, w in enumerate(word_missing):
                            align_word_list.append(('{' + w + '}', (prev_end + i * step,
                                                                    prev_end + (i + 1) * step)))
                        word_missing = []
                    align_word_list.append((word['word'], (cur_start, cur_end)))
        return {'align_word_list': align_word_list, 'num_word_aligned': num_word_aligned}

    def estimate_shift_clip(self, audio_path, audio_start, transcript, offset2time):
        """
        Given an audio clip, call gentle and then estimate a rough mis-alignment
        """
        aligned_clip = self.gentle_solve(audio_path, transcript)

        align_word_list = []
        shift_list = []
        for word_idx, word in enumerate(aligned_clip):
            if word['case'] == 'success':
                if word['startOffset'] in offset2time:
                    shift = word['start'] + audio_start - offset2time[word['startOffset']]
                    if np.abs(shift) <= self.win_size:
                        shift_list.append(shift)
#                 else:
#                     shift = 0
#                 align_word_list.append((word['word'], word['start'] + audio_start, shift))
#         print(align_word_list)
        l = len(shift_list)
        if l < 4:
            return None
        else:
            return np.average(shift_list[l // 4:l * 3 // 4])

    def estimate_shift_window(self, win_idx):
        """
        Estimate rough mis-alignment given a specific window
        """
        transcript, offset2time = self.extract_transcript(
            (win_idx - 1) * self.win_size, (win_idx + 2) * self.win_size, offset_to_time=True)
        shift_list = []
        for audio_shift in range(self.seg_length // 2, self.win_size, self.seg_length):
            audio_start = win_idx * self.win_size + audio_shift
            audio_path = self.extract_audio(audio_start, audio_start + self.clip_length)
            shift = self.estimate_shift_clip(audio_path, audio_start, transcript, offset2time)
            if not shift is None:
                shift_list.append(shift)
        if len(shift_list) == 0:
            return 0
        else:
            shift_list.sort()
            return np.median(shift_list)

    def estimate_shift_all(self):
        """
        Estimate rough mis-alignment for all windows 
        """
        pool = multiprocessing.Pool(self.num_thread)
        shift_window_list = pool.map(self.estimate_shift_window,
                                     [i for i in range(self.num_window)])
        shift_seg_list = []
        for shift in shift_window_list:
            shift_seg_list += [shift] * (self.win_size // self.seg_length)
        shift_seg_list += [shift_seg_list[-1]] * (self.num_seg - len(shift_seg_list))
        return shift_seg_list

    def run_all(self):
        """
        Entrance for solving transcript-audio alignment
        """
        self.load_transcript(self.transcript_path)

        if self.estimate:
            self.shift_seg_list = self.estimate_shift_all()
            print("Estimating rough shift done")

        self.extract_audio_all()
        print("Extracting audio done")

        self.extract_transcript_all()
        print("Extracting transcripts done")

        pool = multiprocessing.Pool(self.num_thread)
        self.result_all = pool.map(self.align_segment_thread, [i for i in range(self.num_seg)])

        align_word_list = []
        num_word_aligned = 0
        for seg_idx, seg in enumerate(self.result_all):
            align_word_list += [word for word in seg['align_word_list']]
            num_word_aligned += seg['num_word_aligned']

        print('num_word_total: ', self.num_words)
        print('num_word_aligned: ', num_word_aligned)
        print('word_missing by total words: ', 1 - 1. * num_word_aligned / self.num_words)
        print('word_missing by total aligned: ', 1 - 1. * num_word_aligned / len(align_word_list))

        if not self.align_dir is None:
            output_path = os.path.join(self.align_dir, self.video_name + '.word.srt')
            TranscriptAligner.dump_aligned_transcript_byword(align_word_list, output_path)
#             output_path = os.path.join(self.align_dir, self.video_name + '.align.srt')
#             TranscriptAligner.dump_aligned_transcript(align_word_list, output_path)
        return {'word_missing': 1 - 1. * num_word_aligned / self.num_words}

#----------Modified methods for scanner pipeline---------------

    def run_segment_scanner(self, audio_pkg, caption_pkg):
        def extract_transcript(caption, start, end, offset_to_time=False):
            if offset_to_time:
                offset2time = {}
            else:
                offset2punc = []
            text_total = ''
            caption_cat = []
            for cap in caption:
                caption_cat += cap
            for cap in caption_cat:
                text, ss, ee = cap['line'], cap['start'], cap['end']
                if ss >= end:
                    break
                if ss >= start:
                    offset = len(text_total)
                    if offset_to_time:
                        words = text.replace('.', ' ').replace('-', ' ').split(' ')
                        step = (ee - ss) / len(words)
                        for i, w in enumerate(words):
                            offset2time[offset] = ss + i * step
                            offset += len(w) + 1
                    else:
                        for p in self.punctuation_all:
                            pp = p.replace('[', '').replace(']', '')
                            for match in re.finditer(p, text):
                                offset2punc.append((match.start() + offset, pp))
                    text_total += text + ' '
            if offset_to_time:
                return text_total, offset2time
            else:
                offset2punc.sort()
                return text_total, offset2punc

        def extract_audio(audio, seg_type=None, clip=None):
            audio_shift = self.audio_rate * self.audio_shift
            if clip is None:
                if seg_type == 'left':
                    audio_tmp = np.concatenate((audio[1], audio[2][:audio_shift, ...]), 0)
                elif seg_type == 'right':
                    audio_tmp = np.concatenate((audio[0][-audio_shift:, ...], audio[1]), 0)
                elif seg_type == 'middle':
                    audio_tmp = np.concatenate(
                        (audio[0][-audio_shift:, ...], audio[1], audio[2][:audio_shift, ...]), 0)
            else:
                audio_tmp = audio[self.audio_rate * clip[0]:self.audio_rate * clip[1]]
            audio_path = tempfile.NamedTemporaryFile(suffix='.wav').name
            wavf.write(audio_path, self.audio_rate, audio_tmp)
            return audio_path

        # entrance
        self.audio_rate = audio_pkg[0].shape[0] // self.seg_length
        if not self.estimate:
            # get center seg index
            if len(caption_pkg[1]) == 0:
                return {'align_word_list': [], 'num_word_aligned': 0, 'num_word_total': 0}
            else:
                seg_idx = int(caption_pkg[1][0]['start'] // self.seg_length)
            left_bound = 1 if (audio_pkg[0][:100] == audio_pkg[1][:100]).all() else 0
            right_bound = 1 if (audio_pkg[1][:100] == audio_pkg[2][:100]).all() else 2
            if left_bound != 0:
                seg_type = 'left'
            elif right_bound != 2:
                seg_type = 'right'
            else:
                seg_type = 'middle'
            audio_path = extract_audio(audio_pkg, seg_type=seg_type)
            transcript, punctuation = extract_transcript(
                caption_pkg[left_bound:right_bound + 1],
                seg_idx * self.seg_length - self.text_shift,
                (seg_idx + 1) * self.seg_length + self.text_shift,
                offset_to_time=False)
            result_seg = self.align_segment(seg_idx, audio_path, transcript, punctuation)
            # count total number of words
            num_words = 0
            for line in caption_pkg[1]:
                for w in line['line'].replace('.', ' ').replace('-', ' ').split():
                    num_words += 1 if w.islower() or w.isupper() else 0
            result_seg['num_word_total'] = num_words
            return result_seg
        else:
            num_seg_in_win = int(self.win_size // self.seg_length)  #5
            seg_center_rel = num_seg_in_win + num_seg_in_win // 2  #7
            # get center seg
            is_empty_window = True
            for s in range(num_seg_in_win // 2):
                if len(caption_pkg[seg_center_rel + s]) > 0:
                    seg_center_abs = int(
                        caption_pkg[seg_center_rel + s][0]['start'] // self.seg_length) - s
                    is_empty_window = False
                    break
                if len(caption_pkg[seg_center_rel - s]) > 0:
                    seg_center_abs = int(
                        caption_pkg[seg_center_rel - s][0]['start'] // self.seg_length) + s
                    is_empty_window = False
                    break
            if is_empty_window or seg_center_abs % num_seg_in_win != num_seg_in_win // 2:
                return {'align_word_list': [], 'num_word_aligned': 0, 'num_word_total': 0}
            win_idx = seg_center_abs // num_seg_in_win
            # get left right boundary
            left_bound = right_bound = None
            for s in range(seg_center_rel):  # error
                if left_bound is None and (audio_pkg[seg_center_rel - s][:100] ==
                                           audio_pkg[seg_center_rel - s - 1][:100]).all():
                    left_bound = seg_center_rel - s
                if right_bound is None and (audio_pkg[seg_center_rel + s][:100] ==
                                            audio_pkg[seg_center_rel + s + 1][:100]).all():
                    right_bound = seg_center_rel + s
            left_bound = 0 if left_bound is None else left_bound
            right_bound = len(audio_pkg) - 1 if right_bound is None else right_bound
            # sample clip to estimate shift
            transcript, offset2time = extract_transcript(
                caption_pkg[left_bound:right_bound + 1], (win_idx - 1) * self.win_size,
                (win_idx + 2) * self.win_size,
                offset_to_time=True)
            shift_list = []
            for s in range(-(num_seg_in_win // 2), num_seg_in_win // 2 + 1):
                seg_rel = seg_center_rel + s
                if seg_rel < left_bound or seg_rel > right_bound:
                    continue
                seg_abs = seg_center_abs + s
                audio_start = seg_abs * self.seg_length + self.seg_length // 2
                audio_path = extract_audio(
                    audio_pkg[seg_rel],
                    clip=(self.seg_length // 2, self.seg_length // 2 + self.clip_length))
                clip_shift = self.estimate_shift_clip(audio_path, audio_start, transcript,
                                                      offset2time)
                if not clip_shift is None:
                    shift_list.append(clip_shift)
            if len(shift_list) == 0:
                win_shift = 0
            else:
                shift_list.sort()
                win_shift = np.median(shift_list)
            # align all segments inside the window
            result_win = {'align_word_list': [], 'num_word_aligned': 0, 'num_word_total': 0}
            for s in range(-(num_seg_in_win // 2), num_seg_in_win // 2 + 1):
                seg_rel = seg_center_rel + s
                if seg_rel < left_bound or seg_rel > right_bound:
                    continue
                seg_abs = seg_center_abs + s
                if seg_rel == left_bound:
                    seg_type = 'left'
                elif seg_rel == right_bound:
                    seg_type = 'right'
                else:
                    seg_type = 'middle'
                audio_path = extract_audio(audio_pkg[seg_rel - 1:seg_rel + 2], seg_type=seg_type)
                transcript, punctuation = extract_transcript(
                    caption_pkg[left_bound:right_bound + 1],
                    seg_abs * self.seg_length - self.text_shift - win_shift,
                    (seg_abs + 1) * self.seg_length + self.text_shift - win_shift,
                    offset_to_time=False)
                result_seg = self.align_segment(seg_abs, audio_path, transcript, punctuation)
                result_win['align_word_list'] += result_seg['align_word_list']
                result_win['num_word_aligned'] += result_seg['num_word_aligned']
                # count total number of words
                num_words = 0
                for line in caption_pkg[seg_rel]:
                    for w in line['line'].replace('.', ' ').replace('-', ' ').split():
                        num_words += 1 if w.islower() or w.isupper() else 0
                result_win['num_word_total'] += num_words
            return result_win

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


#-------------Scanner kernel for transcript alignment--------------
#### First run ####
@scannerpy.register_python_op(name='AlignTranscript', stencil=[-1, 0, 1])
#### Second run ####
# @scannerpy.register_python_op(name='AlignTranscript', stencil=[-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])
class AlignTranscript(Kernel):
    def __init__(self, config):
        win_size = config.args['win_size']
        seg_length = config.args['seg_length']
        max_misalign = config.args['max_misalign']
        num_thread = config.args['num_thread']
        estimate = config.args['estimate']
        self.aligner = TranscriptAligner(
            win_size=win_size,
            seg_length=seg_length,
            max_misalign=max_misalign,
            num_thread=num_thread,
            estimate=estimate)

    def new_stream(self, args):
        #         print(args['video_name'])
        pass

    def execute(self, audio: Sequence[FrameType], captions: Sequence[bytes]) -> bytes:
        cap_json = [json.loads(cap.decode('utf-8')) for cap in captions]
        result_seg = self.aligner.run_segment_scanner(audio, cap_json)
        return pickle.dumps(result_seg)


#------- Scanner pipeline for transcript alignment--------------
class AlignTranscriptPipeline(Pipeline):
    #### First run ####
    job_suffix = 'align_transcript'
    #### Second run ####
    # job_suffix = 'align_transcript2'

    base_sources = ['audio', 'captions']
    run_opts = {
        'pipeline_instances_per_node': 8,
        'io_packet_size': 4,
        'work_packet_size': 4,
        'checkpoint_frequency': 5
    }
    custom_opts = ['align_opts']
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)

    #     parser_fn = lambda _: self.parse
    #     def parse(self, buf, config):
    #         return pickle.loads(buf)

    def build_pipeline(self):
        return {
            #### First run ####
            'align_transcript':
            #### Second run ####
            # 'align_transcript2':
            self._db.ops.AlignTranscript(
                audio=self._sources['audio'].op,
                captions=self._sources['captions'].op,
                win_size=self._custom_opts['align_opts']['win_size'],
                seg_length=self._custom_opts['align_opts']['seg_length'],
                max_misalign=self._custom_opts['align_opts']['max_misalign'],
                num_thread=self._custom_opts['align_opts']['num_thread'],
                estimate=self._custom_opts['align_opts']['estimate'])
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
                '{}_{}'.format(arg['path'], self.job_suffix) for arg in self._sources['audio'].args
            ])


align_transcript_pipeline = AlignTranscriptPipeline.make_runner()
