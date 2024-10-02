def compute_fwt(tot_train_videos, clip_file):
    '''
    clip_file is a textual file of the form:
    0:clip_value
    1:clip_value
    ...
    
    Where 0, 1, etc. is the number of training videos seen so far.
    '''
    
    if tot_train_videos <= 1:
        raise ValueError("[ERROR] The total number of training videos must be > 1!")
    
    const = 1 / (tot_train_videos - 1)
    avg_clip_init = 0.0
    summation = 0.0
    
    with open(clip_file, "r") as file:
        lines = file.readlines()
        
        for line in lines:
            video_n, avg_clip_current = line.strip().split(':')
            if video_n == 0:
                avg_clip_init = avg_clip_current
            
            if video_n >= 2:
                summation += (avg_clip_current - avg_clip_init)
        
        file.close()
    
    fwt = const * summation
    return fwt