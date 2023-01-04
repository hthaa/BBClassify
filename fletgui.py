import flet as ft

def main(page: ft.Page):
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)
    page.update()

    page.window_height = 768
    page.window_width = 665
    page.title = "BB-Classify"

    filepath = ft.TextField(width = 470, text_align=ft.TextAlign.LEFT, label = "Data file path.", hint_text = "Drive:\\Folder\...", tooltip = "Enter file-path to a plain-text file (e.g., .txt, .csv) with a single column of values (no text!) representing final test-scores.")
    choosefile = ft.FilledButton(text = "Choose file.", width = 150, tooltip = "Choose plain-text file (e.g., .txt, .csv) with a single column of values representing final test-scores. Decimal points must be marked with a period ('.').", on_click = lambda _: file_picker.pick_files(allow_multiple = False))
    approach = ft.Dropdown(width = 630, label="Approach.", hint_text = "Choose estimation method.", tooltip = "The Hanson and Brennan approach requires items to be scored as integers. The Livingston and Lewis approach does not require the items to be scored in a particular manner.", 
    options = [ft.dropdown.Option("Hanson and Brennan."), ft.dropdown.Option("Livingston and Lewis.")])
    min_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Minimum possible score.", hint_text = "0", tooltip = "The minimum score that it is possible to attain on the test. Only required for the Livingston and Lewis approach (assumed to be 0 for the Hanson and Brennan approach.)")
    max_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Maximum possible score.", hint_text = "0", tooltip = "The maximum score that it is possible to attain on the test.")
    cut_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Cut-point(s).", hint_text = "0, 0, 0", tooltip = "The cut-points marking the thresholds for categorization. If there two or more cut-points, seperate each cut-point with a comma (',').")
    reliability = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Reliability coefficient.", hint_text = "0.00", tooltip = "The test-score reliability coefficient (e.g., Cronbach's alpha). It is recommended that this value is specified down to at least the third decimal place.")
    c1 = ft.Checkbox(label= "Model parameter estimates.", value = True, tooltip= "Whether the output is to include the Beta-Binomial model parameter estimates.")
    c2 = ft.Checkbox(label= "Model fit test.", value = True, tooltip = "Whether to conduct and report a model-fit test as part of the output.")
    c3 = ft.Checkbox(label= "Accuracy estimates.", value = True, tooltip = "Whether to conduct and report classification accuracy analysis.")
    c4 = ft.Checkbox(label= "Consistency estimates.", value = True, tooltip= "Whether to conduct and report classification consistency analysis.")

    page.add(
        ft.Text("Input:"),
        ft.Row(controls = [filepath, choosefile]),        
        approach,
        ft.Row(
                controls = [min_number, max_number, cut_number, reliability]
            ),
        ft.Divider(), 
        ft.Text("Output:"),
        c1, c2, c3, c4,
        ft.Divider()           
    )


    page.update()

ft.app(target = main)

