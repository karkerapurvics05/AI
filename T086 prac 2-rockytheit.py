dict_hn = {
    'Andheri Station': 10,
    'DN Nagar': 7,
    'Western Express Highway': 6,
    'Juhu Circle': 4,
    'JVPD': 2,
    'PVR Dynamix Juhu Mall': 0
}
dict_gn = dict(
    **{
        'Andheri Station': dict(
            **{
                'DN Nagar': 3,
                'Western Express Highway': 4
            }
        ),
        'DN Nagar': dict(
            **{
                'Juhu Circle': 4
            }
        ),
        'Western Express Highway': dict(
            **{
                'Juhu Circle': 5,
                'JVPD': 6
            }
        ),
        'Juhu Circle': dict(
            **{
                'JVPD': 2,
                'PVR Dynamix Juhu Mall': 5
            }
        ),
        'JVPD': dict(
            **{
                'PVR Dynamix Juhu Mall': 2
            }
        ),
        'PVR Dynamix Juhu Mall': dict()
    }
)
import queue as Q
start = 'Andheri Station'
goal = 'PVR Dynamix Juhu Mall'
result = ''
def get_fn(citystr):
    cities = citystr.split(" , ")
    hn = gn = 0

    for ctr in range(0, len(cities) - 1):
        gn = gn + dict_gn[cities[ctr]][cities[ctr + 1]]

    hn = dict_hn[cities[len(cities) - 1]]

    return (hn + gn)
def expand(cityq):
    global result

    tot, citystr, thiscity = cityq.get()

    if thiscity == goal:
        result = citystr + " : : " + str(tot)
        return

    for cty in dict_gn[thiscity]:
        cityq.put(
            (
                get_fn(citystr + " , " + cty),
                citystr + " , " + cty,
                cty
            )
        )
    expand(cityq)
def main():
    cityq = Q.PriorityQueue()

    thiscity = start

    cityq.put(
        (
            get_fn(start),
            start,
            thiscity
        )
    )
    expand(cityq)
    print("Purvi.R.Karkera T086 :: The A* path with the total is:")
    print(result)
main()
