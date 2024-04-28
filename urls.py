from django.urls import path
from .views import (
    PostListView,
    PostDetailView,
    PostCreateView,
    PostUpdateView,
    PostDeleteView,
    UserPostListView,
    uploadImg,
    ImageCreate,
    show_images,
)
from . import views
from django.contrib.auth.models import User
# urls.py


urlpatterns = [
    path('', views.uploadImg, name='blog-home'),#
    path('user/<str:username>', UserPostListView.as_view(), name='user-posts'),#
    path('post/<int:pk>/', PostDetailView.as_view(), name='post-detail'),#
    path('post/new/', PostCreateView.as_view(), name='post-create'),
    path('post/<int:pk>/update/', PostUpdateView.as_view(), name='post-update'),
    path('media/Files/<int:pk>', PostDeleteView.as_view(), name='media-post-delete'),
    path('media/Files/<int:pk>',PostDeleteView.as_view(),name='post-delete' ),
    path('search/',views.search,name='search' ),
    path('about/', views.about, name='blog-about'),#
    path('upload_user',views.uploadImg,name='blog-uploadImg'),#
    path('create',views.ImageCreate,name='blog-ImgCreate'),#
    path('imagecreated/<str:username>',views.show_images,name='blog-ImgShow'),#
    path('imagecreated/temps/lst/<str:image_id>.png', views.image_view, name='image_view'),#
    path('get_number/', views.get_number, name='get_number'),
    path('process_description/', views.process_description, name='process_description'),
     path('', views.chat, name='chat'),
    path('get_ai_response/', views.get_ai_response, name='get_ai_response'),
  
]
