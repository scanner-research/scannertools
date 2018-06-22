from .prelude import *
from .audio import Audio


class Video:
    """
    Reference to a video file on disk.

    Currently only supports mp4.
    """

    def __init__(self, path, scanner_name=None):
        """
        Args:
            path (str): Path to video file
        """

        self._path = path
        self._decoder_handle = None
        self._scanner_name = scanner_name

    # Lazily load decoder
    def _decoder(self):
        if self._decoder_handle is None:
            try:
                video_file = storehouse.RandomReadFile(get_storage(), self._path.encode('ascii'))
            except UserWarning:
                raise Exception('Path to video `{}` does not exist.'.format(self._path))
            self._decoder_handle = hwang.Decoder(video_file)
        return self._decoder_handle

    def path(self):
        """
        Returns:
            str: Video file path.
        """
        return self._path

    def scanner_name(self):
        """
        Returns:
            str: Name of the video file in the Scanner database.
        """

        return self._scanner_name or self.path()

    def width(self):
        """
        Returns:
            int: Width in pixels of the video.
        """
        return self._decoder().video_index.frame_width()

    def height(self):
        """
        Returns:
            int: Height in pixels of the video.
        """
        return self._decoder().video_index.frame_height()

    def fps(self):
        """
        Returns:
            float: Frames per seconds of the video.
        """
        return self._decoder().video_index.fps()

    def num_frames(self):
        """
        Returns:
            int: Number of frames in the video.
        """
        return self._decoder().video_index.frames()

    def duration(self):
        """
        Returns:
            int: Length of the video in seconds.
        """
        return self._decoder().video_index.duration()

    def frame(self, number=None, time=None):
        """
        Extract a single frame from the video into memory.

        Exactly one of number or time should be specified.

        Args:
            number (int, optional): The index of the frame to access.
            time (float, optional): The time in seconds of the frame to access.

        Returns:
            np.array: (h x w x 3) np.uint8 image.
        """

        if time is not None:
            return self.frames(times=[time])[0]
        else:
            return self.frames(numbers=[number])[0]

    def frames(self, numbers=None, times=None):
        """
        Extract multiple frames from the video into memory.

        Args:
            numbers (List[int], optional): The indices of the frames to access.
            times (List[float], optional): The times in seconds of the frames to access.

        Returns:
            List[np.array]: List of (h x w x 3) np.uint8 images.
        """
        if times is not None:
            numbers = [int(n * self.fps()) for n in times]

        return self._decoder().retrieve(numbers)

    def audio(self):
        """
        Extract the audio from the video.

        Returns:
            Audio: Reference to the audio file.
        """
        audio_path = ffmpeg_extract(input_path=self.path(), output_ext='.wav')
        return Audio(audio_path)

    def extract(self, path=None, ext='.mp4', segment=None):
        """
        Extract an mp4 out of the video.

        Args:
            path (str, optional): Path to write the video.
            ext (str, optional): Video extension to write
            segment (Tuple(int, int), optional): Start/end in seconds

        Returns:
            str: Path to the created video
        """

        return ffmpeg_extract(
            input_path=self.path(), output_path=path, output_ext=ext, segment=segment)

    def montage(self, frames, rows=None, cols=None):
        """
        Create a tiled montage of frames in the video.

        Args:
            frames (List[int]): List of frame indices.
            rows (List[int], optional): Number of rows in the montage.
            cols (List[int], optional): Number of columns in the montage.

        Returns:
            np.array: Image of the montage.
        """

        frames = self.frames(frames)
        return tile(frames, rows=rows, cols=cols)
