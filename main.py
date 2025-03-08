import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from captcha.image import ImageCaptcha
import numpy as np
import string
import random
from keras.models import load_model
import cv2

characters = string.digits + string.ascii_letters+' '

def find_split_points(projection):
    # 找到垂直投影中的低谷
    min_val = np.min(projection)
    max_val = np.max(projection)
    min_threshold = min_val + (max_val - min_val) * 0.1
    max_threshold=max_val-(max_val-min_val)*0.7
    split_points = []
    in_split = 0
    start_index = 0
    
    for i in range(len(projection)):
        if projection[i] >= min_threshold and not in_split:
            in_split = 1
            start_index = i
        elif projection[i] >= max_threshold and in_split:
            in_split = 2
        elif projection[i] < min_threshold and in_split == 1:
            in_split = 0
        elif projection[i] < min_threshold and in_split == 2:
            in_split = 0
            split_points.append((start_index, i))
    #分割多了，只取前四个
    while split_points.__len__()>4:
        argmin=0
        min=split_points[0][1]-split_points[0][0]
        for i in range(len(split_points)):
            if split_points[i][1]-split_points[i][0]<min:
                min=split_points[i][1]-split_points[i][0]
                argmin=i
        split_points.pop(argmin)
    

    return split_points

def cal_projection(L_image):
    _, binary = cv2.threshold(L_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    projection=binary.sum(axis=0)

    return projection


def resize_image(image, target_size):
    # 计算目标大小
    target_width, target_height = target_size
    
    # 获取当前图像的大小
    current_height, current_width, _ = image.shape

    if current_width > target_width:
        # 计算裁剪量
        crop_amount = current_width - target_width
        left_crop = crop_amount // 2
        right_crop = crop_amount - left_crop
        image = image[:, left_crop:-right_crop, :]
        current_width = target_width
    # 计算填充量
    top_padding = (target_height - current_height) // 2
    bottom_padding = target_height - current_height - top_padding
    left_padding = (target_width - current_width) // 2
    right_padding = target_width - current_width - left_padding
    
    # 使用填充方法调整图像大小
    padded_image = cv2.copyMakeBorder(
        image,
        top_padding, bottom_padding,
        left_padding, right_padding,
        cv2.BORDER_CONSTANT,
        value=[1, 1, 1]
    )
    
    return padded_image

def recognize_one_character(image,model,start,end):
    x = np.zeros((1, 50, 50, 3), dtype=np.float32)
    max_pred,max_ans,max_s,max_e = 0,'',start,end
    start_r=[start-8,start-6,start-4,start-2,start,start+2]
    end_r=[end-2,end,end+2,end+4,end+6,end+8]
    for s in start_r:
        for e in end_r:
            x[0] = resize_image(image[:, max(s,0):min(e,200), :], (50, 50))
            pred = model.predict(x[0].reshape(1, 50, 50, 3),verbose=0)
            if np.max(pred) > max_pred and np.argmax(pred) != len(characters) - 1:
                max_pred = np.max(pred)
                max_ans = characters[np.argmax(pred)]
                max_s=max(s,0)
                max_e=min(e,200)


    return max_pred,max_ans,max_s,max_e

def inference(image, split_points,model):
    ans = ['']*4
    if split_points.__len__() ==4:
        for i,(start, end) in enumerate(split_points):
            max_pred,max_ans,_,_=recognize_one_character(image,model,start,end)
            ans[i]=max_ans
            print(f"Char {i+1}: {max_pred}, {max_ans}")
  
    else:        
        i=0
        argmax=0
        max=split_points[0][1]-split_points[0][0]
        for j in range(len(split_points)):
            if split_points[j][1]-split_points[j][0]>max:
                max=split_points[j][1]-split_points[j][0]
                argmax=j
        for ind,(start,end) in enumerate(split_points):
            char_num=1
            if split_points.__len__()==1:
                char_num=4
            elif split_points.__len__()==2 and abs((split_points[ind-1][1]-split_points[ind-1][0])-(end-start))>40 and ind==argmax:
                char_num=3
            elif (split_points.__len__()==3 and ind==argmax) or \
                (split_points.__len__()==2 and abs((split_points[ind-1][1]-split_points[ind-1][0])-(end-start))<=40):
                char_num=2
            s=start
            e=end
            print(f"char_num:{char_num}")
            for _ in range(char_num):
                e=s+(end-start)//char_num
                max_pred,max_ans,max_s,max_e=recognize_one_character(image,model,s,e)
                ans[i]=max_ans
                i+=1
                print(f"Char {i}: {max_pred}, {max_ans},({max_s},{max_e})")

                s=max_e
                    
                    

    return ans

if __name__ == '__main__':
    acc=0
    for i in range(100):
        captcha_text = ''.join([random.choice(characters[:-1]) for _ in range(4)])
        image=ImageCaptcha(200,50).generate_image(captcha_text)
        L_image=image.convert('L')

        kernel = np.ones((2, 2), np.uint8)
        denoised_image = cv2.morphologyEx(np.array(L_image), cv2.MORPH_OPEN, kernel)
        projection=cal_projection(denoised_image)

        split_points = find_split_points(projection)
        print("Split Points:", split_points)
        image=np.array(image)/255.0
        cv2.imwrite(f'captcha_set1/captcha{i}.png',image*255)

        model=load_model('Best_Captcha.h5')
        
        ans=inference(image,split_points,model)
        print(ans)
        print(list(captcha_text))
        ans=''.join(ans)
        if ans.upper()==captcha_text.upper():
            acc+=1
            with open('captcha_set1/record.txt','a') as f:
                f.write(f"{i}:{split_points},true\n")
                f.write(f"{i}:{ans}\n")
                f.write(f"{i}:{list(captcha_text)}\n\n")
        else:
            #将image保存至captcha_set1文件夹下
            with open('captcha_set1/record.txt','a') as f:
                f.write(f"{i}:{split_points},false\n")
                f.write(f"{i}:{ans}\n")
                f.write(f"{i}:{list(captcha_text)}\n\n")

        print(f"acc:{acc},total:{i+1}")
