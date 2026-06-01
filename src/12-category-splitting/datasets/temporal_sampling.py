def loadvideo_decord(self, sample):
    """Load video content using Decord"""
    fname = sample

    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        if self.keep_aspect_ratio:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        else:
            vr = VideoReader(fname, width=self.width, height=self.height,
                                num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []

    if self.mode == 'eval':
        all_index = []
        average_duration = float(len(vr) - 1) / float(self.num_frames)
        interval_cross_view = average_duration / self.eval_num_segments
        for i in range(self.num_frames):
            for ci in range(self.eval_num_segments):
                start = int(np.round(average_duration * i + interval_cross_view * ci))
                end = int(np.round(average_duration * i + interval_cross_view * (ci + 1)))
                all_index.append(int((start + end) // 2))
        
        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer
    else:
        # for train, val and edit, randomly sample frames
        all_index = []
        average_duration = float(len(vr) - 1) / self.num_frames
        for i in range(self.num_frames):
            start = int(np.round(average_duration * i))
            end = int(np.round(average_duration * (i + 1)))
            all_index.append(int(np.random.randint(start, end + 1)))

        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer
