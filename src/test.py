import forecast as fc

def test(usegpu):
    forecast = fc.forecast(usegpu=usegpu)
    while 1:
        rawnews = input("输入新闻:")
        ans = forecast.get(rawnews)
        print(ans)