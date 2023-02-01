
import requests
import bs4


def search(query):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"
    }
    url = f"https://google.com/search?q={query}"
    res = requests.get(url, headers=headers)
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    headings = soup.find_all("h3")
    headings_text = [info.getText() for info in headings]

    if "Description" in headings_text:
        response = headings[headings_text.index("Description")].find_parent().find("span").text

    elif "About featured snippets" in soup.body.text:
        response = soup.body.find(string="About featured snippets").find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find("div").find_all("span")[0].text

    else:
        response = [x.find_parent().find_parent().find_parent().find_parent().find_parent().find_all("span")[-1].text for x in headings][0]

    return response
