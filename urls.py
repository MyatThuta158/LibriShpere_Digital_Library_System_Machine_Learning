from django.urls import path
from .views import predict_new_members, predict_membership_subscriptions
from .chatbot_views import chat, chat_health, chat_rebuild

urlpatterns = [
    # Prediction endpoints
    path('predict/users', predict_new_members, name='predict_new_members'),
    path('predict/subscriber', predict_membership_subscriptions, name='predict_membership_subscriptions'),

    # Chatbot endpoints
    path('chat', chat, name='chat'),
    path('chat/health', chat_health, name='chat_health'),
    path('chat/rebuild', chat_rebuild, name='chat_rebuild'),
]