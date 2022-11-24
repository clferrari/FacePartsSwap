import cv2
import face_alignment
import h5py
import numpy as np
import scipy.io
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from Matrix_operations import Matrix_op, Vector_op
from _3DMM import _3DMM
from evaluate import evaluate
from util_for_graphic import graphic_tools
from scipy.spatial import Delaunay


# Get the face detector
face_detector_module = __import__('face_alignment.detection.sfd', globals(), locals(), ['sfd'], 0)
face_detector = face_detector_module.FaceDetector(device='cpu', verbose=False)
# 3DMM fitting regularization
_lambda = 0.05
# 3DMM fitting rounds (default 1)
_rounds = 1
# Parameters for the HPR operator (do not change)
_r = 3
_C_dist = 700
# Frontalized image rendering step: smaller -> higher resolution image
_rendering_step = 0.5
# Params dict
params3dmm = {'lambda': _lambda, 'rounds': _rounds, 'r': _r, 'Cdist': _C_dist}
# Instantiate all objects
_3DMM_obj = _3DMM()
_graph_tools_obj = graphic_tools(_3DMM_obj)
# Landmark detector
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')
# Load 3DMM components and weights file
components_DL_300 = h5py.File('3D_data/components_DL_300.mat', 'r')
# Setup correct structures
Components = np.transpose(np.array(components_DL_300["Components"]))
Weights = np.transpose(np.array(components_DL_300["Weights"]))
Components_res = np.transpose(np.array(components_DL_300["Components_res"]))
# Setup 3D objects
m_X_obj = Matrix_op(Components, None)
m_X_obj.reshape(Components)
v_weights_obj = Vector_op(Weights)
# Load 3D Model Data
avgModel = np.load('3D_data/avgModel.npy')
idx_landmarks_3D = np.load('3D_data/idxLandmarks3D.npy')
landmarks_3D = np.load('3D_data/landmarks3D.npy')


avgModel_sub = scipy.io.loadmat('3D_data/avgModel_sub/avgModel_sub394_idx.mat')
idxm = np.transpose(np.array(avgModel_sub["midx"]))
idxm = list(map(int, idxm))

avgModel_sub_2 = h5py.File('3D_data/avgModel_sub/avgModel_sub5_idx.mat', 'r')
idxm22 = np.array(avgModel_sub_2["idxm"])
idxm22 = list(map(int, idxm22))
idxm2 = [idxm22[i]-1 for i, el in enumerate(idxm22)]

face_indices_contour = h5py.File('3D_data/parts_indices_contour3.mat', 'r')

tri = h5py.File('3D_data/Tnew.mat', 'r')
Tnew = np.array(tri["Tnew"])
Tnew = Tnew - 1

# Compute triangulation for downsampled 3DMM
triDel = Delaunay(avgModel[idxm, :2])
tri1 = triDel.simplices

triDel = Delaunay(avgModel[idxm2, :2])
tri2 = triDel.simplices

# 3dmm object with data
dict3dmm = {'compObj': m_X_obj, 'weightsObj': v_weights_obj,
            'avgModel': avgModel, 'idx_landmarks_3D': idx_landmarks_3D,
            'landmarks3D': landmarks_3D}
# Image size after resize
im_size = 512


def deformAndResample3DMM(obj3dmm, dict3dmm, lm, params):
    estimation = obj3dmm.opt_3DMM_fast(dict3dmm['weightsObj'].V, dict3dmm['compObj'].X_after_training,
                                       dict3dmm['compObj'].X_res,
                                       dict3dmm['landmarks3D'],
                                       dict3dmm['idx_landmarks_3D'],
                                       lm,
                                       dict3dmm['avgModel'],
                                       params['lambda'],
                                       params['rounds'],
                                       params['r'],
                                       params['Cdist'])
    deformed_mesh = estimation["defShape"]
    projected_mesh = np.transpose(
        obj3dmm.getProjectedVertex(deformed_mesh, estimation["S"], estimation["R"], estimation["T"]))
    return estimation, projected_mesh


def detect_landmarks(im, detector):
    # Swap RGB BGR
    im = im[..., ::-1]

    lm = detector.get_landmarks(im)
    lm = np.delete(lm[0], (65, 51), axis=0)
    return lm


def enlarge_bbox(x, y, w, h, percx, percy):
    xc = x + np.floor(w / 2)
    yc = y + np.floor(h / 2)
    dim = np.floor(np.max([w, h]) / 2)

    xnew = int(np.floor(xc - (dim * percx)))
    ynew = int(np.floor(yc - (dim * percy)))
    wnew = np.floor(dim * percx) * 2
    hnew = np.floor(dim * percx) * 2
    ynew_bott = int(ynew + hnew)
    xnew_bott = int(xnew + wnew)

    # Check if not over borders
    xnew = np.max([xnew, 0])
    ynew = np.max([ynew, 0])

    return xnew, ynew, xnew_bott, ynew_bott


def make_square_img(img):
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    m = max(old_image_height, old_image_width)
    new_image_width = m+10
    new_image_height = m+10
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img

    return result


def make_square_bbox(bbox, resize_fact=2):
    dr = 10
    cx = bbox[0] + bbox[2] // 2
    cy = bbox[1] + bbox[3] // 2
    cr = max(bbox[2], bbox[3]) // 2

    r = cr + (resize_fact * dr)

    x = cx - r
    y = cy - r
    w = (cx + r) - x
    h = (cy + r) - y

    bbox_square = (x, y, w, h)

    return bbox_square


def crop_img(im):
    # Scale the bbox

    d = face_detector.detect_from_image(im[..., ::-1].copy())
    dd = [x[-1:] for x in d]
    d_src = d[np.argmax(dd)]

    # Width
    d_src[2] = d_src[2] - d_src[0]
    # Height
    d_src[3] = d_src[3] - d_src[1]

    d_src = make_square_bbox(d_src)
    d_src = enlarge_bbox(d_src[0], d_src[1], d_src[2], d_src[3], 1.5, 1.5)

    # Crop image
    cropped_img = im[int(d_src[1]):int(d_src[1]) + int(d_src[3]), int(d_src[0]):int(d_src[0]) + int(d_src[2])]

    return cropped_img


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


# Return true if the index is in the region of the face
def is_inside_face(index, projected_mesh_src, projected_mesh_tgt, face_eval_src, face_eval_tgt):
    # Get the coordinates of the index
    x_src = int(projected_mesh_src[index, 0])
    y_src = int(projected_mesh_src[index, 1])
    x_tgt = int(projected_mesh_tgt[index, 0])
    y_tgt = int(projected_mesh_tgt[index, 1])

    # Check if the coordinates are in a valid region of the face
    if x_src <= im_size and y_src <= im_size and x_tgt <= im_size and y_tgt <= im_size:
        is_inside_src = face_eval_src[y_src][x_src] != [0, 0, 0]
        is_inside_tgt = face_eval_tgt[y_tgt][x_tgt] != [0, 0, 0]
        return is_inside_src.any() and is_inside_tgt.any()

    return False


# Get the index inside a polygon
def get_inside_index(border_indexes, projected_mesh, landmarks):
    # Get coordinates of the border of area
    area_pts = []
    for index in border_indexes:
        x = int(projected_mesh[index, 0])
        y = int(projected_mesh[index, 1])
        area_pts.append((x, y))

    # Create polygon
    area_polygon = Polygon(area_pts)

    # Check if points are inside polygon
    inside_indexes = []
    for index in landmarks:
        x = int(projected_mesh[index, 0])
        y = int(projected_mesh[index, 1])
        p = Point(x, y)
        if area_polygon.contains(p):
            inside_indexes.append(index)

    return inside_indexes


def fill_black_pix(img1, img2):
    # Minor correction: if the two masks do not overlap, complete missing pixels with original image
    black_pixels = np.where(
        (img1[:, :, 0] < 1) &
        (img1[:, :, 1] < 1) &
        (img1[:, :, 2] < 1)
    )
    img1[black_pixels] = img2[black_pixels]
    # ---------------------------
    return img1


# Swap part of face of two subject
def face_swap(img_path, img2_path, *parts_to_swap, visDebug = False, cropImg = False):
    # Load, crop and resize source image
    img = cv2.imread(img_path)
    img = make_square_img(img)

    if cropImg:
        img = crop_img(img)

    if img.shape[0] != im_size:
        img = cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA)

    if visDebug:
        cv2.imwrite('debug/source.jpg', img)

    # Detect landmarks
    print("Detecting landmarks Source Image...")
    lm_src = detect_landmarks(img, fa)

    # Deform 3DMM
    print("Deforming 3DMM Source Image...")
    estimation_src, projected_mesh_src = \
        deformAndResample3DMM(_3DMM_obj, dict3dmm, lm_src[:, :2], params3dmm)

    # Create mask
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    # Parsing source face
    print('Segmenting Source Image')
    face_eval_src = evaluate(img, 'res/vis_results', '79999_iter.pth', use_cpu=True)
    face_eval_src = face_eval_src.astype(np.uint8)

    if visDebug:
        cv2.imwrite('debug/mask_src.jpg', face_eval_src)

    # Load, crop and resize target image
    img2 = cv2.imread(img2_path)
    img2 = make_square_img(img2)
    if cropImg:
        img2 = crop_img(img2)
    if img2.shape[0] != im_size:
        img2 = cv2.resize(img2, (im_size, im_size), interpolation=cv2.INTER_AREA)

    if visDebug:
        cv2.imwrite('debug/target.jpg', img2)

    print("Detecting landmarks Target Image...")
    lm_dst = detect_landmarks(img2, fa)

    print("Deforming 3DMM Target Image...")
    estimation_tgt, projected_mesh_tgt = \
        deformAndResample3DMM(_3DMM_obj, dict3dmm, lm_dst[:, :2], params3dmm)

    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Parsing target image
    print('Parsing Target Image')
    face_eval_tgt = evaluate(img2, 'res/vis_results', '79999_iter.pth', use_cpu=True)
    face_eval_tgt = face_eval_tgt.astype(np.uint8)

    if visDebug:
        cv2.imwrite('debug/mask_tgt.jpg', face_eval_tgt)

    # Create new image placeholder
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    # List of uppercase parts to swap
    parts_to_swap_upper = []
    for part in parts_to_swap:
        parts_to_swap_upper.append(part.upper())

    # Indexes to swap
    indexes_to_swap = []

    if np.in1d('FACE', parts_to_swap_upper):
        indexes_to_swap = idxm
    else:
        for part in parts_to_swap_upper:
            if part == 'MOUTH':
                mouth_indexes = np.transpose(np.array(face_indices_contour["mouth"])).astype(int)
                mouth_indexes -= 1
                indexes_to_swap.extend(get_inside_index(mouth_indexes.squeeze(), projected_mesh_src, idxm2))
            elif part == 'NOSE':
                nose_indexes = np.transpose(np.array(face_indices_contour["nose"])).astype(int)
                nose_indexes -= 1
                indexes_to_swap.extend(get_inside_index(nose_indexes.squeeze(), projected_mesh_src, idxm2))
            elif part == 'EYES':
                left_eye_indexes = np.transpose(np.array(face_indices_contour["leye"])).astype(int)
                right_eye_indexes = np.transpose(np.array(face_indices_contour["reye"])).astype(int)
                left_eye_indexes -= 1
                right_eye_indexes -=1
                indexes_to_swap.append(get_inside_index(left_eye_indexes.squeeze(), projected_mesh_src, idxm2))
                indexes_to_swap.append(get_inside_index(right_eye_indexes.squeeze(), projected_mesh_src, idxm2))
            elif part == 'EYEBROWS':
                right_eyebrows_indexes = np.load('3D_data/index/reyebrowIndex.npy')
                left_eyebrows_indexes = np.load('3D_data/index/leyebrowIndex.npy')
                indexes_to_swap.append(get_inside_index(left_eyebrows_indexes, projected_mesh_src, idxm2))
                indexes_to_swap.append(get_inside_index(right_eyebrows_indexes, projected_mesh_src, idxm2))
            elif part == 'LEFT_EYE':
                left_eye_indexes = np.transpose(np.array(face_indices_contour["leye"])).astype(int)
                left_eye_indexes -= 1
                indexes_to_swap.extend(get_inside_index(left_eye_indexes.squeeze(), projected_mesh_src, idxm2))
            elif part == 'RIGHT_EYE':
                right_eye_indexes = np.transpose(np.array(face_indices_contour["reye"])).astype(int)
                right_eye_indexes -= 1
                indexes_to_swap.extend(get_inside_index(right_eye_indexes.squeeze(), projected_mesh_src, idxm2))
            elif part == 'LEFT_EYEBROW':
                left_eyebrows_indexes = np.load('3D_data/index/leyebrowIndex.npy')
                indexes_to_swap.extend(get_inside_index(left_eyebrows_indexes.squeeze(), projected_mesh_src, idxm2))
            elif part == 'RIGHT_EYEBROW':
                right_eyebrows_indexes = np.load('3D_data/index/reyebrowIndex.npy')
                indexes_to_swap.extend(get_inside_index(right_eyebrows_indexes.squeeze(), projected_mesh_src, idxm2))

    # Create tmp image to save debug
    img_tmp = cv2.imread(img_path)
    img_tmp2 = cv2.imread(img2_path)
    if cropImg:
        img_tmp = crop_img(img_tmp)
        img_tmp2 = crop_img(img_tmp2)
    if img_tmp.shape[0] != im_size:
        img_tmp = cv2.resize(img_tmp, (im_size, im_size), interpolation=cv2.INTER_AREA)
        img_tmp2 = cv2.resize(img_tmp2, (im_size, im_size), interpolation=cv2.INTER_AREA)

    if not indexes_to_swap:
        print("NO PARTS TO SWAP")
        return

    # Face 1
    landmarks_points = []
    # Indexes accepted in first face
    accepted_indexes = []

    if len(indexes_to_swap) > 2: # meaning there is a single list
        indexes_to_swap = [indexes_to_swap]

    for idx_to_swap in indexes_to_swap:
        lm_points = []
        # Indexes accepted in first face
        accepted_idx = []
        for index in idx_to_swap:
            # Points visible in both faces
            if np.in1d(index, estimation_src['visIdx']) and np.in1d(index, estimation_tgt['visIdx']):
                x = int(projected_mesh_src[index, 0])
                y = int(projected_mesh_src[index, 1])
                if is_inside_face(index, projected_mesh_src, projected_mesh_tgt, face_eval_src, face_eval_tgt):
                    accepted_idx.append(index)
                    lm_points.append((x, y))
                    cv2.circle(img_tmp, (x, y), 1, (0, 0, 255), -1)

        landmarks_points.append(lm_points)
        accepted_indexes.append(accepted_idx)

    if visDebug:
        cv2.imwrite('debug/points_on_src.jpg', img_tmp)

    convexhull = []
    convexhull2 = []
    for lm_points, accepted_idx in zip(landmarks_points, accepted_indexes):
        indexes_triangles = []
        # Create convex hull of points to locate area to swap
        points = np.array(lm_points, np.int32)
        convhull = cv2.convexHull(points)

        convexhull.append(convhull)
        # Combine with mask obtained with segmentation
        cv2.fillConvexPoly(mask, convhull, 255)
        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        if visDebug:
            cv2.imwrite("debug/cropped_src.jpg", face_image_1)

        # Delaunay triangulation
        print("DELAUNAY TRIANGULATION...")
        rect = cv2.boundingRect(convhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(lm_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

        # Face 2
        landmarks_points2 = []

        for index in accepted_idx:
            x = int(projected_mesh_tgt[index, 0])
            y = int(projected_mesh_tgt[index, 1])
            landmarks_points2.append((x, y))
            cv2.circle(img_tmp2, (x, y), 1, (0, 0, 255), -1)

        points2 = np.array(landmarks_points2, np.int32)
        convhull2 = cv2.convexHull(points2)
        convexhull2.append(convhull2)

        if visDebug:
            cv2.imwrite('debug/points_on_tgt.jpg', img_tmp2)

        # Triangulation of both faces
        print("TRIANGULATION OF BOTH FACES...")
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = lm_points[triangle_index[0]]
            tr1_pt2 = lm_points[triangle_index[1]]
            tr1_pt3 = lm_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1

            if visDebug:
                cv2.line(img_tmp, tr1_pt1, tr1_pt2, (255, 255, 255), 1)
                cv2.line(img_tmp, tr1_pt2, tr1_pt3, (255, 255, 255), 1)
                cv2.line(img_tmp, tr1_pt1, tr1_pt3, (255, 255, 255), 1)

            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)

            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Triangulation of second face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            if visDebug:
                cv2.line(img_tmp2, tr2_pt1, tr2_pt2, (255, 255, 255), 1)
                cv2.line(img_tmp2, tr2_pt2, tr2_pt3, (255, 255, 255), 1)
                cv2.line(img_tmp2, tr2_pt1, tr2_pt3, (255, 255, 255), 1)

            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    if visDebug:
         cv2.imwrite('debug/triang_src.jpg', img_tmp)
         cv2.imwrite('debug/triang_tgt.jpg', img_tmp2)

    # Face swapped (putting 1st face into 2nd face)
    convhull2 = [item for chull2 in convexhull2 for item in chull2]

    img2_face_mask = np.zeros_like(img2_gray)
    for chull2 in convexhull2:
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, np.array(chull2), 255)

    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)

    result = cv2.add(img2_head_noface, img2_new_face)

    # Minor correction: if the two masks do not overlap, complete missing pixels with original image
    result = fill_black_pix(result, img2)

    (x, y, w, h) = cv2.boundingRect(np.array(convhull2))
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    # Create seamless clone and save it
    seamless_clone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    median = cv2.medianBlur(seamless_clone, 3)

    return median, seamless_clone
