import re

from django.template import Library
from django.template.defaultfilters import stringfilter
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe

from numpy import array, int64
array([]);int64()  # Keep the import

register = Library()


@stringfilter
def spacify(value, autoescape=None):
    """https://stackoverflow.com/questions/721035/django-templates-stripping-spaces"""
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
    return mark_safe(re.sub('\s', '&' + 'nbsp;', esc(value)))


spacify.needs_autoescape = True
register.filter(spacify)

@stringfilter
def tablify_report(report: str):
    report = eval(report)
    t = """<table class="center">
        <tr>
            <th>Member</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
        """
    for name, metrics in report.items():
        if name == "micro avg":  # insert blank row before statistics
            classstring = "class=\"top-border\""
        else:
            classstring = ""
        if metrics['support'] > 0:
            t += f"""<tr {classstring}>
                <td><b>{name}</b></td>
                <td>{metrics['precision']}</td>
                <td>{metrics['recall']}</td>
                <td>{metrics['f1-score']}</td>
                <td>{metrics['support']}</td>
            </tr>
            """
    t += "\n</table>\n"
    return mark_safe(t)


register.filter(tablify_report)


@stringfilter
def tablify_confusion(confusion):
    confusion: dict = eval(confusion)
    names = confusion.keys()
    t = """<table class="center"><tr><th></th>"""
    for n in names:
        t += """<th class="conf-header">{}</th>""".format(n)
    t += "</tr>"
    for idx, (name, arr) in enumerate(confusion.items()):
        t += f"""<tr><td><b>{name}</b></td>"""
        for idxx, val in enumerate(arr):
            if idx == idxx:
                t += f"""<td><b>{val}</b></td>"""
            else:
                t += f"""<td>{val}</td>"""
        t += "</tr>"
    t += "</table>"
    return mark_safe(t)


register.filter(tablify_confusion)
