import opendata_connect

try:
    if 'apple' in os.environ['BUILD']:
        root_dir = '../gc_opendata-samples'
    else:
        root_dir = 'E:\gc_opendata'
except:
    root_dir = 'E:\gc_opendata'

od = opendata_connect.open_dataset(root_dir)

od.show_athlete_ids()

ov = od.get_athlete_summary(od.athlete_ids[2])

print(ov.tail(5))