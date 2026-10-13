"""Date-first output names in the user's Moscow date convention."""
from datetime import datetime,timezone,timedelta
import re

def dated_name(title,date=None):
    if not title or any(c in title for c in '/\\:'):
        raise ValueError('Use a simple output filename stem')
    if re.match(r'^\d{4}_\d{2}_\d{2}_',title):return title
    if date is None:date=datetime.now(timezone(timedelta(hours=3))).strftime('%Y_%m_%d')
    else:datetime.strptime(date,'%Y_%m_%d')
    return date+'_'+title
