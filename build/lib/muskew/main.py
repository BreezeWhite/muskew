import os
import sys
import pickle
import argparse
from PIL import Image

import cv2
import numpy as np
import scipy.ndimage
import onnxruntime as rt
from scipy.interpolate import griddata, interp1d
from sklearn.linear_model import LinearRegression


MODULE_PATH = os.path.abspath(os.path.dirname(__file__))


def resize_image(image):
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    w, h = image.size
    pis = w * h
    if 3000000 <= pis <= 435000:
        return image
    lb = 3000000 / pis
    ub = 4350000 / pis
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    print(f"Resized to: {tar_w} x {tar_h} (original: {w} x {h})")
    return image.resize((tar_w, tar_h))


def inference(img_path, step_size=128, batch_size=16, manual_th=None):
    model_path = os.path.join(MODULE_PATH, "checkpoint")
    onnx_path = os.path.join(model_path, "model.onnx")
    metadata = pickle.load(open(os.path.join(model_path, "metadata.pkl"), "rb"))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = rt.InferenceSession(onnx_path, providers=providers)
    output_names = metadata['output_names']
    input_shape = metadata['input_shape']
    output_shape = metadata['output_shape']

    # Collect data
    # Tricky workaround to avoid random mistery transpose when loading with 'Image'.
    image = cv2.imread(img_path)
    image = Image.fromarray(image).convert("RGB")
    image = np.array(resize_image(image))
    win_size = input_shape[1]
    data = []
    for y in range(0, image.shape[0], step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            hop = image[y:y+win_size, x:x+win_size]
            data.append(hop)

    # Predict
    pred = []
    for idx in range(0, len(data), batch_size):
        print(f"{idx+1}/{len(data)} (step: {batch_size})", end="\r")
        batch = np.array(data[idx:idx+batch_size])
        out = sess.run(output_names, {'input': batch})[0]
        pred.append(out)

    # Merge prediction patches
    output_shape = image.shape[:2] + (output_shape[-1],)
    out = np.zeros(output_shape, dtype=np.float32)
    mask = np.zeros(output_shape, dtype=np.float32)
    hop_idx = 0
    for y in range(0, image.shape[0], step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            batch_idx = hop_idx // batch_size
            remainder = hop_idx % batch_size
            hop = pred[batch_idx][remainder]
            out[y:y+win_size, x:x+win_size] += hop
            mask[y:y+win_size, x:x+win_size] += 1
            hop_idx += 1

    out /= mask
    if manual_th is None:
        class_map = np.argmax(out, axis=-1)
    else:
        assert len(manual_th) == output_shape[-1]-1, f"{manual_th}, {output_shape[-1]}"
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(out[..., idx+1]>th, 1, 0)

    return class_map, out


class Grid:
    def __init__(self):
        self.id: int = None
        self.bbox: list[int] = None  # XYXY
        self.y_shift: int = 0

    @property
    def y_center(self):
        return (self.bbox[1]+self.bbox[3]) / 2

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


class GridGroup:
    def __init__(self):
        self.id: int = None
        self.reg_id: int = None
        self.bbox: list[int] = None
        self.gids: list[int] = []
        self.split_unit: int = None

    @property
    def y_center(self):
        return round((self.bbox[1]+self.bbox[3]) / 2)

    def __lt__(self, tar):
        # Sort by width
        w = self.bbox[2] - self.bbox[0]
        tw = tar.bbox[2] - tar.bbox[0]
        return w < tw

    def __repr__(self):
        return f"Grid Group {self.id} / Width: {self.bbox[2]-self.bbox[0]} / BBox: {self.bbox}" \
            f" / Y-center: {self.y_center} / Reg. ID: {self.reg_id}"


def build_grid(st_pred, split_unit=11):
    grid_map = np.zeros(st_pred.shape) - 1
    h, w = st_pred.shape

    is_on = lambda data: np.sum(data) > split_unit//2

    grids = []
    for i in range(0, w, split_unit):
        cur_y = 0
        last_y = 0
        cur_stat = is_on(st_pred[cur_y, i:i+split_unit])
        while cur_y < h:
            while cur_y < h and cur_stat == is_on(st_pred[cur_y, i:i+split_unit]):
                cur_y += 1
            if cur_stat and (cur_y-last_y < split_unit):
                # Switch off
                grid_map[last_y:cur_y, i:i+split_unit] = len(grids)
                gg = Grid()
                gg.bbox = (i, last_y, i+split_unit, cur_y)
                gg.id = len(grids)
                grids.append(gg)
            cur_stat = not cur_stat
            last_y = cur_y
    return grid_map, grids


def build_grid_group(grid_map, grids):
    regions, feat_num = scipy.ndimage.label(grid_map+1)
    grid_groups = []
    for i in range(feat_num):
        region = grid_map[regions==i+1]
        gids = list(np.unique(region).astype(int))
        gids = sorted(gids)
        lbox = grids[gids[0]].bbox
        rbox = grids[gids[-1]].bbox
        box = (
            min(lbox[0], rbox[0]),
            min(lbox[1], rbox[1]),
            max(lbox[2], rbox[2]),
            max(lbox[3], rbox[3]),
        )
        gg = GridGroup()
        gg.reg_id = i + 1
        gg.gids = gids
        gg.bbox = box
        gg.split_unit = lbox[2] - lbox[0]
        grid_groups.append(gg)

    grid_groups = sorted(grid_groups, reverse=True)
    gg_map = np.zeros_like(regions) - 1
    for idx, gg in enumerate(grid_groups):
        gg.id = idx
        gg_map[regions==gg.reg_id] = idx
        gg.reg_id = idx

    return gg_map, grid_groups


def connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids, ref_count=8, max_step=20):
    new_gg_map = np.copy(gg_map)
    ref_gids = grid_groups[0].gids[:ref_count]
    idx = 0
    gg = grid_groups[idx]
    remaining = set(range(len(grid_groups)))
    while remaining:
        # Check if not yet visited
        if gg.id not in remaining:
            if remaining:
                gid = remaining.pop()
                gg = grid_groups[gid]
                ref_gids = gg.gids[:ref_count]
            else:
                break
        else:
            remaining.remove(gg.id)

        if len(ref_gids) < 2:
            continue

        # Extend on the left side
        step_size = gg.split_unit
        centers = [grids[gid].y_center for gid in ref_gids]
        x = np.arange(len(centers)).reshape(-1, 1) * step_size
        model = LinearRegression().fit(x, centers)
        ref_box = grids[ref_gids[0]].bbox

        end_x = ref_box[0]
        h = ref_box[3] - ref_box[1]
        cands_box = []  # Potential trajectory
        for i in range(max_step):
            tar_x = (-i - 1) * step_size
            cen_y = model.predict([[tar_x]])[0]  # Interpolate y center
            y = int(round(cen_y - h / 2))
            region = new_gg_map[y:y+h, end_x-step_size:end_x]  # Area to check
            unique, counts = np.unique(region, return_counts=True)
            labels = set(unique)  # Overlapped grid group IDs
            if -1 in labels:
                labels.remove(-1)

            cands_box.append((end_x-step_size, y, end_x, y+h))
            if len(labels) == 0:
                end_x -= step_size
            else:
                cands_box = cands_box[:-1]  # Remove the overlapped box

                # Determine the overlapped grid group id
                if len(labels) > 1:
                    # Overlapped with more than one group
                    overlapped_size = sorted(zip(unique, counts), key=lambda it: it[1], reverse=True)
                    label = overlapped_size[0][0]
                else:
                    label = labels.pop()

                # Check the overlappiong with the traget grid group is valid.
                tar_box = grid_groups[label].bbox
                if tar_box[2] > end_x:
                    break

                # Start assign grid to disconnected position.
                # Get the grid ID.
                yidx, xidx = np.where(region==label)
                yidx += y
                xidx += end_x-step_size
                reg = grid_map[yidx, xidx]
                grid_id = np.unique(reg)
                assert len(grid_id) == 1, grid_id
                assert grid_id in grid_groups[label].gids, f"{grid_id}, {label}"
                grid = grids[int(grid_id[0])]

                # Interpolate y centers between the start and end points again.
                centers = [grid.y_center, centers[0]]
                x = [-i-1, 0]
                inter_func = interp1d(x, centers, kind='linear')

                # Start to insert grids between points
                cands_ids = []
                for bi, box in enumerate(cands_box):
                    interp_y = round(inter_func(-bi-1) - h/2)
                    grid = Grid()
                    box = (box[0], interp_y, box[2], interp_y+h)
                    grid.bbox = box
                    grid.id = len(grids)
                    cands_ids.append(len(grids))
                    gg.gids.append(len(grids))
                    gg.bbox = (
                        min(gg.bbox[0], box[0]),
                        min(gg.bbox[1], box[1]),
                        max(gg.bbox[2], box[2]),
                        max(gg.bbox[3], box[3])
                    )
                    gg.bbox = [int(bb) for bb in gg.bbox]
                    box = [int(bb) for bb in box]
                    grids.append(grid)
                    new_gg_map[box[1]:box[3], box[0]:box[2]] = gg.id

                # Continue to track on the overlapped grid group.
                gg = grid_groups[label]
                gids = gg.gids + cands_ids[::-1]
                ref_gids = gids[:ref_count]

                break

    return new_gg_map


def build_mapping(gg_map, min_width_ratio=0.4):
    regions, num = scipy.ndimage.label(gg_map+1)
    min_width = gg_map.shape[1] * min_width_ratio

    points = []
    coords_y = np.zeros_like(gg_map)
    period = 10
    for i in range(num):
        y, x = np.where(regions==i+1)
        w = np.max(x) - np.min(x)
        if w < min_width:
            continue

        target_y = round(np.mean(y))

        uniq_x = np.unique(x)
        for ii, ux in enumerate(uniq_x):
            if ii % period == 0:
                meta_idx = np.where(x==ux)[0]
                sub_y = y[meta_idx]
                cen_y = round(np.mean(sub_y))
                coords_y[int(target_y), int(ux)] = cen_y
                points.append((target_y, ux))

    # Add corner case
    coords_y[0] = 0
    coords_y[-1] = len(coords_y) - 1
    for i in range(coords_y.shape[1]):
        points.append((0, i))
        points.append((coords_y.shape[0]-1, i))

    return coords_y, np.array(points)


def estimate_coords(staff_pred):
    ker = np.ones((6, 1), dtype=np.uint8)
    pred = cv2.dilate(staff_pred.astype(np.uint8), ker)
    ker = np.ones((1, 6), dtype=np.uint8)
    pred = cv2.erode(pred, ker)

    print("Building grids")
    grid_map, grids = build_grid(pred)

    print("Labeling areas")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    print("Connecting lines")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    print("Building mapping")
    coords_y, points = build_mapping(new_gg_map)

    print("Dewarping")
    vals = coords_y[points[:, 0].astype(int), points[:, 1].astype(int)]
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1]
    coords_y = griddata(points, vals, (grid_x, grid_y), method='linear')

    coords_x = grid_y.astype(np.float32)
    coords_y = coords_y.astype(np.float32)
    return coords_x, coords_y


def get_parser():
    parser = argparse.ArgumentParser(
        "muskew",
        description="Muskew helps deskew/dewarp the curved music sheet photo with AI-aidded approach.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        help="Path to the input image. \
            The output file will also be in this folder, with postfix '_deskew' added.",
        type=str
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    infile = args.infile
    print(f"Received file: {infile}")

    print("Predicting stafflines")
    staff_pred, _ = inference(infile)
    staff_pred = np.where(staff_pred==1, 1, 0)

    print("Deskewing")
    coords_x, coords_y = estimate_coords(staff_pred)
    img = cv2.imread(infile).astype(np.float32)
    img = cv2.resize(img, (staff_pred.shape[1], staff_pred.shape[0]))
    for i in range(img.shape[-1]):
        img[..., i] = cv2.remap(img[..., i], coords_x, coords_y, cv2.INTER_CUBIC)

    print("Writing the result")
    ext = os.path.splitext(infile)[1]
    out_file = infile.replace(ext, "_deskew"+ext)
    cv2.imwrite(out_file, img)
    print(f"File written to: {out_file}")


if __name__ == "__main__":
    main()
