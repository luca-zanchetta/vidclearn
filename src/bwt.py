def compute_bwt(tot_train_videos, clip_file_middle, clip_file_end):
    if tot_train_videos <= 1:
        raise ValueError("[ERROR] The total number of training videos must be > 1!")
    
    const = 1 / (tot_train_videos - 1)
    summation = 0.0
    end_scores = []
    middle_scores = []
    
    with open(clip_file_middle, "r") as file:
        lines = file.readlines()
        for line in lines:
            video_n, clip_current = line.strip().split(':')
            middle_scores.append(clip_current)
        
    with open(clip_file_end, "r") as file:
        lines = file.readlines()
        for line in lines:
            video_n, clip_current = line.strip().split(':')
            end_scores.append(clip_current)
    
    for i in range(1, tot_train_videos):
        summation += (end_scores[i-1] - middle_scores[i-1])
    
    bwt = const * summation
    return bwt