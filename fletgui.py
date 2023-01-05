import flet as ft

def main(page: ft.Page):
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)
    page.update()

    page.window_height = 768
    page.window_width = 665
    page.title = "BB-Classify"

    filepath = ft.TextField(width = 470, text_align=ft.TextAlign.LEFT, label = "Data file path", hint_text = "Drive:\\Folder\...", tooltip = "Enter file-path to a plain-text file (e.g., .txt, .csv) with a single column of values (no text!) representing final test-scores.")
    choosefile = ft.FilledButton(text = "Choose file", width = 150, tooltip = "Choose plain-text file (e.g., .txt, .csv) with a single column of values representing final test-scores. Decimal points must be marked with a period ('.').", on_click = lambda _: file_picker.pick_files(allow_multiple = False))
    approach = ft.Dropdown(width = 630, label="Approach", hint_text = "Choose estimation method.", tooltip = "The Hanson and Brennan approach requires items to be scored as integers. \nThe Livingston and Lewis approach does not require the items to be scored in a particular manner.", 
    options = [ft.dropdown.Option("Hanson and Brennan"), ft.dropdown.Option("Livingston and Lewis")])
    
    min_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Minimum score", hint_text = "0", tooltip = "The minimum score that it is possible to attain on the test. Only required for the Livingston and Lewis approach (assumed to be 0 for the Hanson and Brennan approach).")
    max_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Maximum score", hint_text = "0", tooltip = "For the Livingston and Lewis approach: the maximum score that it is possible to attain on the test. \nFor the Hanson and Brennan approach: the actual test length in terms of number of items.")
    cut_number = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Cut-point(s)", hint_text = "0, 0, 0", tooltip = "The cut-points marking the thresholds for categorization. If there two or more cut-points, seperate each cut-point with a comma (',').")
    reliability = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Reliability coefficient", hint_text = "0.00", tooltip = "The test-score reliability coefficient (e.g., Cronbach's alpha). It is recommended that this value is specified down to at least the third decimal place.")

    model = ft.Dropdown(width = 310, label = "True-score model", hint_text = "Choose model to be fit.", options = [ft.dropdown.Option("Four-parameter"), ft.dropdown.Option("Four-parameter with fail-safe"), ft.dropdown.Option("Two-parameter")], 
    tooltip = "Choose true-score beta-distribution model to be fit.\nThe first option allows location-parameter estimates to be out of bounds.\nThe second fits a two-parameter solution with pre-specified location-parameters if out of bounds.\nThe third fits a two-parameter solution with pre-specified location-parameters.")
    lower_bound = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Lower-bound", hint_text = "0")
    upper_bound = ft.TextField(width = 150, text_align = ft.TextAlign.RIGHT, label = "Upper-bound", hint_text = "1")

    c1 = ft.Checkbox(label = "Model parameter estimates", value = True, tooltip= "Whether the output is to include the Beta-Binomial model parameter estimates.")
    c2 = ft.Checkbox(label = "Model fit test", value = False, tooltip = "Whether to conduct and report a model-fit test as part of the output (yet to be implemented).", disabled = True)
    c3 = ft.Checkbox(label = "Accuracy estimates", value = True, tooltip = "Whether to conduct and report classification accuracy analysis.")
    c4 = ft.Checkbox(label = "Consistency estimates", value = True, tooltip= "Whether to conduct and report classification consistency analysis.")

    submit = ft.FilledButton(width = 150, text = "Submit")

    page.add(
        ft.Text("Input:"),
        ft.Row(controls = [filepath, choosefile]),        
        approach,
        ft.Row(
                controls = [min_number, max_number, cut_number, reliability]
            ),
        ft.Divider(),
        ft.Text("Model fitting controls:"),
        ft.Row(controls = [model, lower_bound, upper_bound]),
        ft.Divider(),
        ft.Text("Output:"),
        c1, c2, c3, c4,
        ft.Divider(),
        submit           
    )


    page.update()

ft.app(target = main)

