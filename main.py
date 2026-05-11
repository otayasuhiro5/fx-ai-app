from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.metrics import dp

class HomeScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = MDBoxLayout(orientation='vertical', padding=dp(16), spacing=dp(12))
        pair_card = MDCard(size_hint_y=None, height=dp(80), padding=dp(16))
        pair_label = MDLabel(text="USD/JPY  149.50", font_style="H5", halign="center")
        pair_card.add_widget(pair_label)
        layout.add_widget(pair_card)
        sign_card = MDCard(size_hint_y=None, height=dp(120), padding=dp(16), md_bg_color=(0.1,0.7,0.3,1))
        sign_label = MDLabel(text="BUY\n信頼度: 85%", halign="center", theme_text_color="Custom", text_color=(1,1,1,1))
        sign_card.add_widget(sign_label)
        layout.add_widget(sign_card)
        tpsl_card = MDCard(size_hint_y=None, height=dp(100), padding=dp(16))
        tpsl_label = MDLabel(text="TP: 150.20\nSL: 149.10", halign="center")
        tpsl_card.add_widget(tpsl_label)
        layout.add_widget(tpsl_card)
        self.add_widget(layout)

class HistoryScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = MDBoxLayout(orientation='vertical', padding=dp(16))
        label = MDLabel(text="シグナル履歴\n（準備中）", halign="center")
        layout.add_widget(label)
        self.add_widget(layout)

class SettingsScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = MDBoxLayout(orientation='vertical', padding=dp(16))
        label = MDLabel(text="設定\n（準備中）", halign="center")
        layout.add_widget(label)
        self.add_widget(layout)

class FXApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Green"
        nav = MDBottomNavigation()
        home_item = MDBottomNavigationItem(name='home', text='ホーム', icon='home')
        home_item.add_widget(HomeScreen())
        nav.add_widget(home_item)
        history_item = MDBottomNavigationItem(name='history', text='履歴', icon='history')
        history_item.add_widget(HistoryScreen())
        nav.add_widget(history_item)
        settings_item = MDBottomNavigationItem(name='settings', text='設定', icon='cog')
        settings_item.add_widget(SettingsScreen())
        nav.add_widget(settings_item)
        return nav

if __name__ == '__main__':
    FXApp().run()
