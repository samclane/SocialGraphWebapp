import re

from django.template import Library
from django.template.defaultfilters import stringfilter
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe

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
