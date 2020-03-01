def yolo_to_xyxy(sz, box):
    x, y, w, h = box
    dw = 1. / sz[0]
    dh = 1. / sz[1]

    w0 = w / dw
    h0 = h / dh
    xmid = x / dw
    ymid = y / dh

    x0, x1 = xmid - w0 / 2., xmid + w0 / 2.
    y0, y1 = ymid - h0 / 2., ymid + h0 / 2.

    return [int(x0), int(y0), int(x1), int(y1)]


def xyxy_to_yolo(sz, box):
    dw = 1. / sz[0]
    dh = 1. / sz[1]
    xmid = (box.x1_int + box.x2_int) / 2.0
    ymid = (box.y1_int + box.y2_int) / 2.0
    w0 = box.x2_int - box.x1_int
    h0 = box.y2_int - box.y1_int
    x = xmid * dw
    y = ymid * dh
    w = w0 * dw
    h = h0 * dh

    return [x, y, w, h]
