from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video("input_data/08fd33_4.mp4")
    # init tracker

    tracker = Tracker("models/best.pt")
    tracker.get_object_tracks(video_frames)
    tracker.adding_game_properties()
    # read_from_stub=False, stub_path="stubs/track_stub.pkl")
    output_video_frames = tracker.draw_annotations()
    save_video(output_video_frames, "output_videos/output_video.avi")









if __name__ == "__main__":
    main()
    
    
    
    

