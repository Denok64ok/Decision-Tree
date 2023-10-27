from prettytable import PrettyTable

global table
table = PrettyTable()
table.field_names = ['Метод', 'Точность на тестовых данных', 'Точность на тренировочных данных']
table.align = "l"
