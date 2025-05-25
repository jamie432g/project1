def get_bbox_width(bbox):
    x1,_,x2,_ = bbox
    return x2-x1

def get_bbox_height(bbox):
    _,y1,_,y2 = bbox
    return y2-y1

def get_centre_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return tuple(map(int, ((x1 + x2) / 2, (y1 + y2) / 2)))

def get_bottom_centre(bbox):
    x1,_,x2,y2 = bbox
    return tuple(map(int,((x1 + x2)/2, y2)))

def get_bottom_centre_float(bbox):
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)
