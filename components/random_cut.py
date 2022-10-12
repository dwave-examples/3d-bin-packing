import numpy as np
from heapq import heapify, heappush, heappop


def random_cut_generator(num_bins, bin_length, bin_width, bin_height, num_cases,
                         seed=111, **kwargs):
    bin_height_ = bin_height / 2
    np.random.seed(seed)
    boxes = [(-max(bin_length, bin_width, bin_height_),
              bin_length, bin_width, bin_height_)] * num_bins
    heapify(boxes)
    c = 0
    while len(boxes) < num_cases and c < 1000:
        b = heappop(boxes)
        i = 1 + np.argmax(b[1:])
        r = 0.5 + (np.random.rand() - 0.5) / 4
        box1 = list(b)
        box2 = list(b)
        cut_length = int(b[i] * r)
        if b[i] == 1 or cut_length == 0 or cut_length >= b[i]:
            c += 1
            heappush(boxes, tuple(b))
            continue
        box1[i] = cut_length
        box2[i] = b[i] - cut_length
        box1[0] = -max(box1[1:])
        box2[0] = -max(box2[1:])
        heappush(boxes, tuple(box1))
        heappush(boxes, tuple(box2))
        c += 1
    _, case_lengths, case_widths, case_heights = zip(*boxes)
    data = {
        "num_bins": num_bins,
        "bin_dimensions": [bin_length, bin_width, bin_height],
        "case_length": case_lengths,
        "case_width": case_widths,
        "case_height": case_heights,
    }
    case_dimensions = np.vstack(
        [data["case_length"], data["case_width"], data["case_height"]]
    )
    unique_dimensions, data["quantity"] = np.unique(case_dimensions,
                                                    axis=1,
                                                    return_counts=True)

    data["case_length"] = unique_dimensions[0, :]
    data["case_width"] = unique_dimensions[1, :]
    data["case_height"] = unique_dimensions[2, :]

    data["case_ids"] = np.array(range(len(data["quantity"])))
    return data
