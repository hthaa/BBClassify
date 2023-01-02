import flet as ft

def main(page: ft.Page):
    page.window_height = 768
    page.window_width = 470 + 150 + 10 + 35
    page.title = "BB-Classify"

    filepath = ft.TextField(width = 470, text_align=ft.TextAlign.LEFT, label = "Data file path.", hint_text = "C:\\...")
    choosefile = ft.FilledButton(text="Choose file.", width = 150)
    approach = ft.Dropdown(width = 470 + 150 + 10, label="Approach.", hint_text="Choose estimation method.", 
    options=[ft.dropdown.Option("Hanson and Brennan."),ft.dropdown.Option("Livingston and Lewis.")])
    min_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Minimum possible score.", hint_text = "0")
    max_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Maximum possible score.", hint_text = "0")
    cut_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Cut-point(s).", hint_text = "0, 0, 0")
    reliability = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Reliability coefficient.", hint_text = "0.00")
    c1 = ft.Checkbox(label="Model parameter estimates.", value = True)
    c2 = ft.Checkbox(label="Model fit test.", value = True)
    c3 = ft.Checkbox(label="Accuracy estimates.", value = True)
    c4 = ft.Checkbox(label="Consistency estimates.", value = True)

    page.add(
        ft.Row(controls = [filepath, choosefile]),        
        approach,
        ft.Row(
                controls = [min_number, max_number, cut_number, reliability]
            ), 
            c1, c2, c3, c4              
    )


    page.update()

ft.app(target = main)

