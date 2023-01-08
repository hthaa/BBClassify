import flet as ft
import scipy.special
from scipy.integrate import quad
from scipy.stats import binom
import statistics as stats
import math
import pandas as pd
import numpy as np
import csv

def etl(mean, var, reliability, min = 0, max = 1):
    return ((mean - min) * (max - mean) - (reliability * var)) / (var * (1 - reliability))
def k(mean, var, reliability, length):
    vare = var * (1 - reliability)
    num = length * ((length - 1) * (var - vare) - length * var + mean * (length - mean))
    den = 2 * (mean * (length - mean) - (var - vare))
    return num / den
def dbeta4p(x, a, b, l, u):
    if x < l or x > u:
        return 0
    else:
        return (1 / scipy.special.beta(a, b)) * (((x - l)**(a - 1) * (u - x)**(b - 1)) / (u - l)**(a + b - 1))
def beta4fit(x, moments = []):
    if len(moments) != 4:
        m1 = stats.mean(x)
        s2 = stats.variance(x)
        x3 = list(x)
        x4 = list(x)
        for i in range(len(x)):
            x3[i] = ((x3[i] - m1)**3) / (s2**0.5)**3
            x4[i] = ((x4[i] - m1)**4) / (s2**0.5)**4
        g3 = (1 / len(x3)) * sum(x3)
        g4 = (1 / len(x4)) * sum(x4)
    else:
        m1 = moments[0]
        s2 = moments[1] - moments[0]**2
        g3 = (moments[2] - 3 * moments[0] * moments[1] + 2 * moments[0]**3) / ((s2**0.5)**3)
        g4 = (moments[3] - 4 * moments[0] * moments[2] + 6 * moments[0]**2 * moments[0] - 3 * moments[0]**3) / ((s2**0.5)**4)
    r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
    if g3 < 0:
        a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    else:
        b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    l = m1 - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    u = m1 + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    return [a, b, l, u]
def beta2fit(x, l, u):
    m1 = stats.mean(x)
    s2 = stats.variance(x)
    a = ((l - m1) * (l * (m1 - u) - m1**2 + m1 * u - s2)) / (s2 * (l - u))
    b = ((m1 - u) * (l * (u - m1) + m1**2 - m1 * u + s2)) / (s2 * (u - l))
    return [a, b, l, u]
def dcbinom(p, N, n, k):
    a = binom.pmf(n, N, p)
    b = binom.pmf(n, N - 2, p)
    c = binom.pmf(n - 1, N - 2, p)
    d = binom.pmf(n - 2, N - 2, p)
    e = k * p * (1 - p)
    return a - e * (b - 2*c + d)
def bbintegrate1(a, b, l, u, N, n, k, lower, upper, method = "ll"):
    if method != "ll":
        def f(x, a, b, l, u, N, n, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n, k))
    else:
        def f(x, a, b, l, u, N, n):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n))
def bbintegrate2(a, b, l, u, N, n1, n2, k, lower, upper, method = "ll"):
    if method != "ll":
        def f(x, a, b, l, u, N, n1, n2, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n1, k) * dcbinom(x, N, n2, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2, k))
    else:
        def f(x, a, b, l, u, N, n1, n2):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n1, N, x) * binom.pmf(n2, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2))
def dfac(x, r):
    x1 = list(x)
    for i in range(len(x)):
        if r <= 1:
            x1[i] = x1[i]**r
        else:
            for j in range(1, r + 1):
                if j > 1:
                    x1[i] = x1[i] * (x[i] - j + 1)
    return x1
def tsm(x, n, k):
    m = [0, 0, 0, 0]
    for i in range(0, 4):
        if i == 0:
            m[i] = stats.mean(x) / n
        else:
            M = i + 1
            a = (dfac([n], 2)[0] + k * dfac([M], 2)[0])
            b = stats.mean(dfac(x, M)) / dfac([n - 2], M - 2)[0]
            c = k * dfac([M], 2)[0] * m[i]
            m[i] = (b / a) + c
    return m
def betaparameters(x, n, k, model = 4, l = 0, u = 1):
    m = tsm(x, n, k)
    s2 = m[1] - m[0]**2
    g3 = (m[2] - 3 * m[0] * m[1] + 2 * m[0]**3) / (math.sqrt(s2)**3)
    g4 = (m[3] - 4 * m[0] * m[2] + 6 * m[0]**2 * m[1] - 3 * m[0]**4) / (math.sqrt(s2)**4)
    if model == 4:
        r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
        if g3 < 0:
            a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
            b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        else:
            b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
            a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        l = m[0] - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
        u = m[0] + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    if model == 2:
        a = ((l - m[0]) * (l * (m[0] - u) - m[0]**2 + m[0] * u - s2)) / (s2 * (l - u))
        b = ((m[0] - u) * (l * (u - m[0]) + m[0]**2 - m[0] * u + s2)) / (s2 * (u - l))
    return {"alpha":  a, "beta": b, "l": l, "u": u}
def cac(x, reliability, min, max, cut, model = 4, l = 0, u = 1, failsafe = False, method = "ll", output = ["parameters", "accuracy", "consistency"]):
    out = {}
    cut = [min] + cut + [max]
    tcut = list(cut)
    for i in range(len(cut)):
        tcut[i] = (tcut[i] - min) / (max - min)
    if isinstance(x, dict):
        pars = x
        if method == "ll":
            N = pars["etl"]
        else:
            N = pars["atl"]
    else:
        if method == "ll":
            Nnotrounded = etl(stats.mean(x), stats.variance(x), reliability, min, max)
            N = round(Nnotrounded)
            pars = betaparameters(x, N, 0, model)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, N, 0, 2, l, u)
            pars["etl"] = Nnotrounded
            pars["etl rounded"] = N
            pars["lords_k"] = 0
            for i in range(len(cut)):
                cut[i] = tcut[i] * N
        else:
            N = max 
            K = k(stats.mean(x), stats.variance(x), reliability, N)
            pars = betaparameters(x, N, K, model, l, u)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, max, N, 2, l, u)
            pars["lords_k"] = K
    if "parameters" in output:
        out["parameters"] = pars    
    if "accuracy" in output:
        confmat = np.zeros((N + 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(N + 1):
                confmat[j, i] = bbintegrate1(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, j, pars["lords_k"], tcut[i], tcut[i + 1], method)[0]
        confusionmatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i != len(cut) - 2:
                    confusionmatrix[i, j] = sum(confmat[cut[i]:cut[i + 1], j])
                else:
                    confusionmatrix[i, j] = sum(confmat[cut[i]:, j])
        accuracy = []
        for i in range(len(cut) - 1):
            accuracy = accuracy + [confusionmatrix[i, i]]
        accuracy = sum(accuracy)
        out["confusionMatrix"] = pd.DataFrame(confusionmatrix)
        out["overallAccuracy"] = accuracy
    
    if "consistency" in output:
        consmat = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                consmat[i, j] = bbintegrate2(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, i, j, pars["lords_k"], 0, 1, method)[0]
        consistencymatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i == 0 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], 0:cut[j + 1]]))
                if i == 0 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1]]))
                if i == 0  and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                if (i != 0 and i != len(cut) - 2) and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], 0:cut[j + 1]]))
                if (i != 0 and i != len(cut) - 2) and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1]]))
                if (i != 0 and i != len(cut) - 2) and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                if i == len(cut) - 2 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, 0:cut[j + 1]]))
                if i == len(cut) - 2 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1]]))
                if i == len(cut) - 2 and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1] + 1]))
            consistency = []
            for i in range(len(cut) - 1):
                consistency = consistency + [consistencymatrix[i, i]]
            consistency = sum(consistency)
            out["consistencyMatrix"] = pd.DataFrame(consistencymatrix)
            out["overallConsistency"] = consistency
    
    return out
def s2n(s):
    out = [int(x) if x.isdigit() else float(x) for x in s.replace(', ', ',').split(',')]
    if any(isinstance(x, float) for x in out):
        out = list(map(float, out))
    if len(out) == 1:
        out = out[0]
    return out
def csv_to_list(x):
    data = []
    with open('test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            row = list(map(float, row))
            data.append(row)
    for i in range(len(data)):
        data[i] = sum(data[i])
    return data

def main(page: ft.Page):
    page.window_height = 825
    page.window_width = 665 * 2
    page.scroll = True
    page.title = "BB-Classify"
        
    def pick_files_result(e: ft.FilePickerResultEvent):
        filepath.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Selection cancelled."            
        )
        filepath.update()

    pick_files_dialog = ft.FilePicker(on_result = pick_files_result)
    page.overlay.append(pick_files_dialog)

    choosefile = ft.FilledButton(
        text = "Choose file...", width = 150, icon = ft.icons.UPLOAD_FILE,
        tooltip = "Choose .csv file where rows represent persons and columns represent items. The rows and columns must not be named (i.e., the file should only consist of numbers representing item-scores). The rows will be summed, so the persons final scores will be their sum-scores. If you wish to employ a different scoring-rule, then perform the scoring manually and supply a .csv file containing only a single column representing final test-scores. Decimal points must be marked with a period ('.').", 
        on_click = lambda _: pick_files_dialog.pick_files(allow_multiple = False)
        )
    
    choosefile.bgcolor = "blue"

    filepath = ft.TextField(width = 470, text_align=ft.TextAlign.LEFT, label = "File name", disabled = True)
    
    approach = ft.Dropdown(
        width = 630, label="Approach", hint_text = "Choose estimation method.", options = [ft.dropdown.Option("Hanson and Brennan"), ft.dropdown.Option("Livingston and Lewis")],
        tooltip = "The Hanson and Brennan approach requires the test-items to be scored as integers. \nThe Livingston and Lewis approach does not require the test-items to be scored in a particular manner."        
        )
    
    min_number = ft.TextField(
        width = 150, text_align = ft.TextAlign.RIGHT, label = "Minimum score", hint_text = "0", 
        tooltip = "The minimum score that it is possible to attain on the test. Only required for the Livingston and Lewis approach (assumed to be 0 for the Hanson and Brennan approach)."
        )
    
    max_number = ft.TextField(
        width = 150, text_align = ft.TextAlign.RIGHT, label = "Maximum score", hint_text = "0", 
        tooltip = "For the Livingston and Lewis approach: the maximum score that it is possible to attain on the test. \nFor the Hanson and Brennan approach: the actual test length in terms of number of items."
        )

    cut_number = ft.TextField(
        width = 150, text_align = ft.TextAlign.RIGHT, label = "Cut-point(s)", hint_text = "0, 0, 0", 
        tooltip = "The cut-points marking the thresholds for categorization. If there are two or more cut-points, seperate each cut-point with a comma (',')."
        )

    reliability = ft.TextField(
        width = 150, text_align = ft.TextAlign.RIGHT, label = "Reliability", hint_text = "0.00", 
        tooltip = "The test-score reliability coefficient (e.g., Cronbach's alpha). It is recommended that this value is specified down to at least the third decimal place."
        )

    model = ft.Dropdown(
        width = 310, label = "True-score model", hint_text = "Choose model to be fit.", 
        options = [ft.dropdown.Option("Four-parameter"), ft.dropdown.Option("Four-parameter with fail-safe"), ft.dropdown.Option("Two-parameter")], 
        tooltip = "Choose true-score beta-distribution model to be fit.\nThe first option allows location-parameter estimates to be out of bounds.\nThe second fits a two-parameter solution with pre-specified location-parameters if out of bounds.\nThe third fits a two-parameter solution with pre-specified location-parameters."
        )
    
    lower_bound = ft.TextField(
        width = 150, text_align = ft.TextAlign.RIGHT, label = "Lower-bound", hint_text = "0", 
        tooltip = "Lower-bound value for the two-parameter model. Must be between 0 and 1, and less than the upper-bound value."
        )

    upper_bound = ft.TextField(
        width = 150, text_align = ft.TextAlign.RIGHT, label = "Upper-bound", hint_text = "1", 
        tooltip = "Upper-bound value for the two-parameter model. Must be between 0 and 1, and greater than the lower-bound value."
        )

    c1 = ft.Checkbox(
        label = "Model parameter estimates", value = True, fill_color = "blue",
        tooltip= "Whether the output is to include the Beta-Binomial model parameter estimates."
        )
    
    c2 = ft.Checkbox(
        label = "Model fit test", disabled = True, value = False, fill_color = "blue",
        tooltip = "Whether to conduct and report a model-fit test as part of the output (yet to be implemented)."
        )
    
    c3 = ft.Checkbox(
        label = "Accuracy estimates", value = True, fill_color = "blue",
        tooltip = "Whether to conduct and report classification accuracy analysis."
        )
    
    c4 = ft.Checkbox(
        label = "Consistency estimates", value = True, fill_color = "blue",
        tooltip = "Whether to conduct and report classification consistency analysis."
        )

    submit = ft.FilledButton(width = 150, text = "Submit")
    submit.bgcolor = "blue"

    md1 = """
    # Markdown Example
    Markdown allows you to easily include formatted text, images, and even formatted Dart code in your app.

    ## Titles

    Setext-style

    #This is an H1 
    =============
    """
    page.fonts = {
        "Roboto Mono": "RobotoMono-VariableFont_wght.ttf",
    }
    results = ft.Markdown(md1, width = 630, selectable = True, code_style = ft.TextStyle(font_family="Roboto Mono"))
    
    page.add(
        ft.Row(controls = [
            ft.Column(controls = [
                ft.Text("Data:"),
                ft.Row(controls = [choosefile, filepath]),        
                ft.Divider(),
                ft.Text("Model fitting controls:"), 
                approach,
                ft.Row(controls = [model, lower_bound, upper_bound]),
                ft.Divider(),
                ft.Text("Test information:"),
                ft.Row(
                    controls = [min_number, max_number, cut_number, reliability]
                ),
                ft.Divider(),
                ft.Text("Output:"),
                c1, c2, c3, c4,
                ft.Divider(),
                submit]
                ),
            ft.Column(controls = [ft.Text("Results:"),
                results
                    ]
                )
            ], vertical_alignment = ft.CrossAxisAlignment.START
        )
        
    )

    page.update()

ft.app(target = main)

