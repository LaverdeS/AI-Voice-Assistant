from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivy.core.text import LabelBase

Window.size = (350, 550)


class Chatbot(MDApp):

    def change_screen(self, name):
        screen_manager.current = name

    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
        # this defines the order of views
        screen_manager.add_widget(Builder.load_file("Chats.kv"))
        screen_manager.add_widget(Builder.load_file("Main.kv"))
        # screen_manager.add_widget(Builder.load_file("Chats.kv"))
        return screen_manager


if __name__ == '__main__':
    LabelBase.register(name="Poppins", fn_regular="Poppins-Regular.ttf")
    Chatbot().run()