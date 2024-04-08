import numpy as np
import matplotlib.pyplot as plt

def visualize_array(arr):
    """
    Visualize a numpy array as an image.
    
    Args:
    - arr (numpy.ndarray): Input numpy array.
    """
    plt.imshow(arr, cmap='gray')  # 'cmap'은 colormap을 설정합니다. 여기서는 흑백 이미지를 위해 'gray'를 사용합니다.
    plt.colorbar()  # 컬러바 추가
    plt.axis('off')  # 축 제거
    plt.title('Visualized Array')
    plt.show()

# numpy 파일 불러오기

for i in range(14, 17):
    file_path = f'/home/myyang/projects/STG-NF/data/ShanghaiTech/gt/test_frame_mask/01_00{i}.npy'
    arr_loaded = np.load(file_path)
    print(arr_loaded)

# 배열 시각화
# visualize_array(arr_loaded)
