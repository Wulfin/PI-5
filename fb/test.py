from facebook_page_scraper import Facebook_scraper

page_list = ['KimKardashian','arnold','joebiden','eminem','SmoshGames','Metallica','cnn']

proxy_port = 10001
posts_count = 100
browser = "firefox"
timeout = 600
headless = False

for page in page_list:
    proxy = f'username:password@us.smartproxy.com:{proxy_port}'
    scraper = Facebook_scraper(page, posts_count, browser, proxy=proxy, timeout=timeout, headless=headless)
    json_data = scraper.scrap_to_json()
    print(json_data)
    filename = page
    scraper.scrap_to_csv(filename, ".")
    proxy_port += 1