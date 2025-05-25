import cv2
import numpy as np
from utils import get_centre_of_bbox, get_bbox_width, get_bottom_centre

def draw_ellipse(frame, bbox, colour, track_id=None):
    _,_,_,y2 = bbox
    y2 = int(y2)
    x_centre, _ = get_centre_of_bbox(bbox)
    bottom_centre = get_bottom_centre(bbox)
    width = get_bbox_width(bbox)
    ellipse_axes = (int(width), int(0.25 * width))

    cv2.ellipse(
        frame,
        center=(bottom_centre),
        axes=ellipse_axes,
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=(0, 0, 0),  
        thickness=4,  
        lineType=cv2.LINE_4
    )

    cv2.ellipse(
        frame,
        center=(bottom_centre),
        axes=ellipse_axes,
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=colour,
        thickness=2,
        lineType=cv2.LINE_4
    )

    if track_id is not None:
        rectangle_width = 30
        rectangle_height = 15
        x1_rect = x_centre - rectangle_width // 2
        x2_rect = x_centre + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 10
        y2_rect = (y2 + rectangle_height // 2) + 10
        rectangle_top_left = (int(x1_rect), int(y1_rect))
        rectangle_bottom_right = (int(x2_rect), int(y2_rect))
        top_left_x,top_left_y = rectangle_top_left

        cv2.rectangle(
            frame,
            rectangle_top_left,
            rectangle_bottom_right,
            colour,
            cv2.FILLED
        )

        cv2.rectangle(
            frame,
            rectangle_top_left,
            rectangle_bottom_right,
            (0,0,0),
            thickness=1
        )


        cv2.putText(
            frame,
            f"{track_id}",
            ((top_left_x + 8), (top_left_y + 12)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    return frame


def draw_triangle(frame, bbox, colour):
    _,y1,_,_ = bbox
    y1 = int(y1)
    x, _ = get_centre_of_bbox(bbox)

    triangle_points = np.array([
        [x, y1],
        [x - 10, y1 - 20],
        [x + 10, y1 - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, colour, thickness= -1)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame
