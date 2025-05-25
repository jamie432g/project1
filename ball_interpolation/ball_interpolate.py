import pandas as pd
class Ball_Interpolate:
    def __init__(self):
        pass
    def interpolate_ball_positions(self, ball_positions):
        #get bbox values of ball
        ball_positions = [entry[1]["bbox"] if 1 in entry and "bbox" in entry[1] else [] for entry in ball_positions]

        #to dataframe
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolation and backfill
        df.interpolate(inplace=True)
        df.bfill(inplace=True)

        #convert ball back to original values
        return [{1: {"bbox": bbox}} for bbox in df.values.tolist()]