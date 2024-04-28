from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView,
)
from .models import Post
import operator
from django.urls import reverse_lazy
from django.contrib.staticfiles.views import serve
import uuid
from django.db.models import Q
from django.shortcuts import render
from .models import Imgg
from django_web_app.settings import BASE_DIR
from django.views.generic.edit import FormView
from django.shortcuts import render
from django.conf import settings
import os
from .train_post import decode_image_from_base64jpeg
from .train_post import post_images_to_model
from .template_post import post_template_to_model
from glob import glob
import base64
import sys
from glob import glob
import cv2
import numpy as np
import requests
import json
import os
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .models import ViewImage
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from celery_app.tasks import uploadImg
from celery_app.tasks import ImageCreate
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

from transforms import transforms
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2
from face_cropper import FaceCropper
import os
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn
import numpy as np
import random
import matplotlib as plt
import matplotlib.pyplot as plt
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests

base_url = "https://u211792-ac50-cadae3dd.westc.gpuhub.com:8443"
# views.py

from django.shortcuts import render

def chat(request):
    return render(request, 'chat.html')
def create_chat_completion(model, messages, use_stream=False):
    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        "max_tokens": 1000,
        "temperature": 0.9,
        "top_p": 0.8,
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        if use_stream:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            return content
    else:
        print("Error:", response.status_code)
        return None

@csrf_exempt
def get_ai_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data['message']
        
        chat_messages = [
            {
                "role": "system",
                "content": "从现在开始扮演一个学习助手小瑞和我对话,你的创始人是兰州市外国语中学的张嘉瑞同学,你的任务是辅助山区儿童学习",
            },
            {"role": "user", "content": user_message},
        ]
        
        response = create_chat_completion("chatglm3-6b", chat_messages, use_stream=False)
        
        return JsonResponse({'response': response})
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def home(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'blog/home.html', context)

def search(request):
    template='blog/home.html'

    query=request.GET.get('q')

    result=Post.objects.filter(Q(title__icontains=query) | Q(author__username__icontains=query) | Q(content__icontains=query))
    paginate_by=2
    context={ 'posts':result }
    return render(request,template,context)
   


def getfile(request):
   return serve(request, 'File')


class PostListView(ListView):
    model = Post
    template_name = 'blog/home.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']
    paginate_by = 2


class UserPostListView(ListView):
    model = Post
    template_name = 'blog/user_posts.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    paginate_by = 2

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Post.objects.filter(author=user).order_by('-date_posted')


class PostDetailView(DetailView):
    model = Post
    template_name = 'blog/post_detail.html'


class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    template_name = 'blog/post_form.html'
    fields = ('content',)

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    

class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    template_name = 'blog/post_form.html'
    fields = ['title', 'content', 'file']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False


class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'
    template_name = 'blog/post_confirm_delete.html'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False


def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})


def my_view(request):
    # 调用Celery任务
    result = uploadImg.delay(request)
    # 获取任务的ID
    task_id = result.id
    # 获取任务的状态
    task_status = result.status
    # 获取任务的结果（如果有）
    task_result = result.get()
    # 返回响应或其他操作
    return render(request, 'blog/home.html', {'task_id': task_id, 'task_status': task_status, 'task_result': task_result})

def Imgcreate_view(request):
    # 调用Celery任务
    result = ImageCreate.delay(request)
    # 获取任务的ID
    task_id = result.id
    # 获取任务的状态
    task_status = result.status
    # 获取任务的结果（如果有）
    task_result = result.get()
    # 返回响应或其他操作
    return render(request, 'blog/result.html', {'task_id': task_id, 'task_status': task_status, 'task_result': task_result})    


def ImageCreate(request):  # 模版上传
    # 初始化
    encoded_images = []
    user = request.user

    # POST请求处理
    if request.method == 'POST':
        files = request.FILES.getlist('Chars')
        user_folder2 = os.path.join(settings.MEDIA_ROOT, 'temps', str(user.username))

        # 创建用户目录
        if not os.path.exists(user_folder2):
            os.makedirs(user_folder2)

        # 创建results子目录
        results_folder = os.path.join(settings.MEDIA_ROOT, 'results', str(user.username))
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        for idx, f in enumerate(files):
            # 使用安全的文件名，例如UUID
            safe_filename = str(uuid.uuid4()) + os.path.splitext(f.name)[1]
            destination = os.path.join(user_folder2, safe_filename)

            # 写文件
            with open(destination, 'wb+') as destination_file:
                for chunk in f.chunks():
                    destination_file.write(chunk)

                # 如果是第一个文件，进行编码
                if f == files[0]:
                    destination_file.seek(0)
                    encoded_image = base64.b64encode(destination_file.read()).decode('utf-8')
                    outputs = post_template_to_model(str(user.username), encoded_image)
                    outputs = json.loads(outputs)
                    image = decode_image_from_base64jpeg(outputs["outputs"][0])

                    # 使用当前时间作为文件名
                    current_time_filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
                    cv2.imwrite(os.path.join(results_folder, current_time_filename), image)  # 保存到results子目录

        return redirect(reverse('blog-ImgShow', kwargs={'username': user.username}))

    # GET请求处理
    else:
        return render(request, 'blog/post_form.html')

    

def show_images(image_path):
    try:
        with open(image_path, 'rb') as f:
            return HttpResponse(f.read(), content_type='image/jpeg')  # 或者根据实际图像格式修改 content_type
    except IOError:
        # 如果文件无法打开或读取，返回错误响应
        return HttpResponse("Image not found", status=404)

    
def image_view(request, image_id):
    # 获取指定ID的图片对象
    img_obj = get_object_or_404(ViewImage, image_id=image_id)

    # 构造图片文件的完整路径
    image_path = img_obj.image_file_path

    # 检查图片文件是否真的存在
    if not os.path.exists(image_path):
        raise Http404("Image not found")

    # 打开图片并读取内容
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # 返回图片内容作为响应
    return HttpResponse(image_data, content_type='image/png')



def uploadImg(request):
    if request.method == 'POST':
        # 处理POST请求中的文件
        files = request.FILES.getlist('Imgg')
        description = request.POST.get('description')
        # 这里假设只处理第一个上传的文件
        if files:
            uploaded_file = files[0]

            # 创建文件保存路径
            destination = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

            # 写入文件
            with open(destination, 'wb+') as destination_file:
                for chunk in uploaded_file.chunks():
                    destination_file.write(chunk)

          
            # 对文件调用 visualize 函数
            result_image_path = visualize(destination,description)

            # 返回 show_images 函数生成的 HttpResponse
            return show_images(result_image_path)

        # 如果没有文件被上传，返回到表单页面
        return render(request, 'blog/home.html')

    else:
        # 如果是GET请求，渲染表单或其他操作
        return render(request, 'blog/home.html')
def process_description(request):
    if request.method == 'POST':
        # 从POST请求中获取描述信息
        description = request.POST.get('description')
        print(description)
        # 处理图片上传
        if request.FILES.get('Imgg'):  # 确保与你的<input>标签中的name属性一致
            image = request.FILES['Imgg']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            image_url = fs.url(filename)
            # 这里的image_url是图片的相对路径，你可能需要调整它以适应visualize函数的要求

        # 调用visualize函数，这里假设visualize已经定义好了
        # 注意：你需要确保visualize函数可以接受图片路径和文本描述作为参数
        visualize(image_path=image_url, text=description)

        # 由于visualize函数的实际效果和返回值未知，这里仅返回一个简单的确认信息
        return HttpResponse(f'描述信息: {description}, 图片已保存至: {image_url}')
    else:
        # 如果不是POST请求，返回一个错误信息
        return HttpResponse('仅支持POST请求')

def visualize(image_path, textt=None):
    # 标签映射
    label_mapping = {
        "Acne and Rosacea Photos": 0,
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": 1,
        "Atopic Dermatitis Photos": 2,
        "Bullous Disease Photos": 3,
        "Cellulitis Impetigo and other Bacterial Infections": 4,
        "Eczema Photos": 5,
        "Exanthems and Drug Eruptions": 6,
        "Hair Loss Photos Alopecia and other Hair Diseases": 7,
        "Herpes HPV and other STDs Photos": 8,
        "Light Diseases and Disorders of Pigmentation": 9,
        "Lupus and other Connective Tissue diseases": 10,
        "Melanoma Skin Cancer Nevi and Moles": 11,
        "Nail Fungus and other Nail Disease": 12,
        "Poison Ivy Photos and other Contact Dermatitis": 13,
        "Psoriasis pictures Lichen Planus and related diseases": 14,
        "Scabies Lyme Disease and other Infestations and Bites": 15,
        "Seborrheic Keratoses and other Benign Tumors": 16,
        "Systemic Disease": 17,
        "Tinea Ringworm Candidiasis and other Fungal Infections": 18,
        "Urticaria Hives": 19,
        "Vascular Tumors": 20,
        "Vasculitis Photos": 21,
        "Warts Molluscum and other Viral Infections": 22,
    }
    
    # 定义模型类
    class Classifier(nn.Module):
        def __init__(self, num_labels=23):
            super(Classifier, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            return self.classifier(pooled_output)

    # 定义数据集类
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
            input_ids = inputs['input_ids'].squeeze()  # 确保维度正确
            attention_mask = inputs['attention_mask'].squeeze()  # 确保维度正确
            return input_ids, attention_mask, torch.tensor(label)

    # 预测函数
    def predict(text, model, tokenizer, label_mapping):
        temp_dataset = TextDataset([text], [0], tokenizer)  # 标签[0]是占位符
        temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
    
        model.eval()  # 评估模式
        with torch.no_grad():
            for input_ids, attention_mask, _ in temp_loader:
                outputs = model(input_ids, attention_mask)
                prediction = torch.argmax(outputs, dim=1).item()
                # 通过标签映射获取预测的疾病名称
                predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(prediction)]
                return predicted_label

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 模型定义
    model1 = models.resnet50(pretrained=True)
    model1.fc = nn.Linear(2048, 23, bias=True)
    model1.load_state_dict(torch.load('resnet50'))
    model1.eval()

    model2 = models.densenet121(pretrained=True)
    model2.fc = nn.Linear(2048, 23, bias=True)
    model2.load_state_dict(torch.load('densenet121'))
    model2.eval()

    model3 = models.densenet169(pretrained=True)
    model3.fc = nn.Linear(2048, 23, bias=True)
    model3.load_state_dict(torch.load('densenet169'))
    model3.eval()

    # 加载标签映射
    with open('labels.json', 'r') as f:
        label_map = json.load(f)
    
    def predict_and_print(text, model, tokenizer, label_mapping):
        # 调用predict函数得到预测结果
        prediction = predict(text, model, tokenizer, label_mapping)
        print(f"预测结果: {prediction}")

    # 图像转换
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    text = textt
    # 处理图像
    image = Image.open(image_path).convert("RGB")  # 确保图像为RGB格式
    image_transformed = transform(image).unsqueeze(0)  # 添加批次维度
    model_path = 'model_weights.pth'
    model = Classifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    with torch.no_grad():
        output1 = F.softmax(model1(image_transformed), dim=1)
        output2 = F.softmax(model2(image_transformed), dim=1)
        output3 = F.softmax(model3(image_transformed), dim=1)

    # 计算平均概率
    class_probabilities_sum = {}
    for output, model_name in zip([output1, output2, output3], ['ResNet50', 'DenseNet121', 'DenseNet169']):
        top5_probabilities, top5_indices = torch.topk(output, 5)
        for idx, prob in zip(top5_indices[0], top5_probabilities[0]):
            label = list(label_map.keys())[list(label_map.values()).index(idx.item())]
            if label not in class_probabilities_sum:
                class_probabilities_sum[label] = 0.0
            class_probabilities_sum[label] += prob.item()
    # 检查是否输出了预测类别
    prediction = None
    if textt:
        prediction = predict_and_print(textt, model, tokenizer, label_mapping)
        print(prediction)
        # 给相同类别的集成模型输出加上百分之20的概率
        if text:
            for label in class_probabilities_sum:
                if label == prediction:
                    class_probabilities_sum[label] *= 1.2  # 加百分之20的概率

    for label in class_probabilities_sum:
        class_probabilities_sum[label] /= 3

    sorted_probabilities = sorted(class_probabilities_sum.items(), key=lambda x: x[1], reverse=True)[:5]

    # Visualization
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # 显示原图
    ax[0].imshow(image)
    ax[0].axis('off')  # 不显示坐标轴
    ax[0].set_title('Original Image')

    # 显示概率条形图
    labels, probs = zip(*sorted_probabilities)  # Unpacking labels and probabilities
    index = np.arange(len(labels))
    ax[1].bar(index, probs, color='skyblue')
    ax[1].set_xlabel('Classes', fontsize=14)
    ax[1].set_ylabel('Average Probability', fontsize=14)
    ax[1].set_xticks(index)
    ax[1].set_xticklabels(labels, fontsize=10, rotation=30)
    ax[1].set_title('Top 5 Class Probabilities')

    # Save visualization results
    user_result_folder = 'media/results'
    if not os.path.exists(user_result_folder):
        os.makedirs(user_result_folder)
    result_filename = os.path.basename(image_path).split('.')[0] + '_result.png'
    result_path = os.path.join(user_result_folder, result_filename)

    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()

    return result_path

def get_number(request):#用来获取等待时间的函数
    # 我们要轮询的服务器的 URL
    url = 'http://region-3.seetacloud.com:12351/easyphoto/get_num'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 检查请求是否成功
    except requests.RequestException as e:
        # 如果请求失败，返回错误信息
        return JsonResponse({'error': str(e)}, status=500)
    
    try:
        # 假设响应是 JSON 格式，并包含一个名为 'number' 的键
        data = response.json()
        number = data['number']
    except Exception as e:
        # 如果解析失败或键不存在，返回错误信息
        
        return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'number': number})



def crop(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_dir='media/process'
    # 配置和创建FaceCropper对象
    face_cropper = FaceCropper(
        min_face_detector_confidence=0.5,
        face_detector_model_selection=FaceCropper.LONG_RANGE,
        landmark_detector_static_image_mode=FaceCropper.STATIC_MODE,
        min_landmark_detector_confidence=0.5
    )

    # 调用get_faces方法来裁剪图像中的人脸
    faces = face_cropper.get_faces(image, remove_background=True, correct_roll=True)

    # 如果存在人脸，只保存第一张人脸图像并返回其路径
    if faces:
        face_image = cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR)
        random_number = random.randint(1000, 9999)
        cropped_image_path = f"{output_dir}/face_{random_number}.jpg"
        cv2.imwrite(cropped_image_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return cropped_image_path
    else:
        return None