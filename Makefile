ROOT_DIR		=	/data/Users/watanabe/CoSPA-dir/U-Net
IDENTIFICATION	=	0
ROOT_RESULT_DIR	=	${ROOT_DIR}/results/${IDENTIFICATION}

# data
DATAROOT_PATH	=	/data/Users/watanabe/Data/Dataset_2q_2022
TRAIN_IMG_DIR	=	${DATAROOT_PATH}/Image/train/
TRAIN_GT_DIR	=	${DATAROOT_PATH}/GT/train/
TEST_IMG_DIR	=	${DATAROOT_PATH}/Image/test/
TEST_GT_DIR		=	${DATAROOT_PATH}/GT/test/

# settings
EPOCH			=	200
TRAIN_IMG_NUM	=	30
GPU_NUM			=	0

# 2d
SRC_DIR_2D		=	${ROOT_DIR}/srcs/2d
RESULT_DIR_2D	=	${ROOT_RESULT_DIR}/2d
MODEL_2D		=	${RESULT_DIR_2D}/model.h5

# 3d
SRC_DIR_3D		=	${ROOT_DIR}/srcs/3d
RESULT_DIR_3D	=	${ROOT_RESULT_DIR}/3d
MODEL_3D		=	${RESULT_DIR_3D}/model.h5
STACK_SIZE		=	16

all: 2d 3d

2d: train_2d test_2d

train_2d: ${MODEL_2D}
${MODEL_2D}:
	python ${SRC_DIR_2D}/train.py \
		--train_img_dir=${TRAIN_IMG_DIR} \
		--train_gt_dir=${TRAIN_GT_DIR} \
		--result_dir=${RESULT_DIR_2D} \
		--epoch=${EPOCH} \
		--train_img_num=${TRAIN_IMG_NUM} \
		--GPU=${GPU_NUM} | tee ${RESULT_DIR_2D}/train.txt

test_2d:
	python ${SRC_DIR_2D}/test.py \
		--test_img_dir=${TEST_IMG_DIR} \
		--result_dir=${RESULT_DIR_2D} | tee ${RESULT_DIR_2D}/test.txt


3d: train_3d test_3d

train_3d: ${MODEL_3D}
${MODEL_3D}:
	python ${SRC_DIR_3D}/train.py \
		--train_img_dir=${TRAIN_IMG_DIR} \
		--train_gt_dir=${TRAIN_GT_DIR} \
		--result_dir=${RESULT_DIR_3D} \
		--epoch=${EPOCH} \
		--train_img_num=${TRAIN_IMG_NUM} \
		--GPU=${GPU_NUM} | tee ${RESULT_DIR_3D}/train.txt

test_3d:
	python ${SRC_DIR_3D}/test.py \
		--test_img_dir=${TEST_IMG_DIR} \
		--result_dir=${RESULT_DIR_3D} \
		--stack_size=${STACK_SIZE} | tee ${RESULT_DIR_3D}/test.txt
