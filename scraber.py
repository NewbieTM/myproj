from bs4 import BeautifulSoup
import requests

HOME_PAGE_URL = 'https://calorizator.ru/product/all'  # Also scrapable
ITERATED_URL = 'https://calorizator.ru/product/all?page='  # 1 -> 84


def get_soup(url):
    """Получает HTML-код страницы и возвращает объект BeautifulSoup."""
    response = requests.get(url)
    response.raise_for_status()  # Проверяем, что запрос успешен
    return BeautifulSoup(response.text, 'html.parser')


def extract_text_from_elements(elements):
    """Извлекает текст из списка элементов."""
    return [element.text.strip() for element in elements]


def get_all_products_from_url(url, products = []):
    """Парсит данные о продуктах с указанной страницы."""
    soup = get_soup(url)

    titles = []
    td_titles = soup.find_all('td', class_='views-field views-field-title active')
    for td in td_titles:
        link = td.find('a')
        if link:
            # print(link.text)
            titles.append(link.text)

    proteins = extract_text_from_elements(soup.find_all('td', class_='views-field views-field-field-protein-value'))
    fat = extract_text_from_elements(soup.find_all('td', class_='views-field views-field-field-fat-value'))
    carbohydrate = extract_text_from_elements(
        soup.find_all('td', class_='views-field views-field-field-carbohydrate-value'))
    kcal = extract_text_from_elements(soup.find_all('td', class_='views-field views-field-field-kcal-value'))

    #print(titles, proteins, fat, carbohydrate, kcal)

    for i, title in enumerate(titles):
        product = {
            'title': title,
            'protein': proteins[i]+'г',
            'fat': fat[i]+'г',
            'carbohydrate': carbohydrate[i]+'г',
            'kcal': kcal[i]+'г'
        }
        products.append(product)

    return products


def main():
    products = get_all_products_from_url(HOME_PAGE_URL)
    for i in range(1,85): # 85
        products = get_all_products_from_url(ITERATED_URL+f'{i}',products)
    for product in products:
        #print(product)
        line = product['title'] + '|' + f'Содержание белков:{product['protein']}\
 жиров:{product['fat']} углеводов:{product['carbohydrate']} калорий:{product['kcal']}'
        with open('products.txt', 'a', encoding='utf-8') as f:
            f.write(line + '\n')

if __name__ == "__main__":
    main()