import xlrd
# from xlwt import Workbook


filePath = "/Users/treasersmac/Programming/LogicRegression-UnitTest/testcases/cases.xlsx"

def readInputs():
    wb = xlrd.open_workbook(filePath)
    sheet1 = wb.sheet_by_index(0)
    col = sheet1.col_values(1)
    return col[1:]

def readExpect():
    wb = xlrd.open_workbook(filePath)
    sheet1 = wb.sheet_by_index(0)
    col = sheet1.col_values(0)
    return col[1:]


if __name__ == "__main__":
    wb = xlrd.open_workbook(filePath)
    sheet1 = wb.sheet_by_index(0)
    rows = sheet1.row_values(0)  # 获取行内容，索引从0开始
    cols0 = sheet1.col_values(0)
    cols1 = sheet1.col_values(1)
    cols2 = sheet1.col_values(1)
    cols3 = sheet1.col_values(1)
    print("rows:", rows, "\ncol0:", cols0, "\ncol1:", cols1, "\ncol2:", cols2, "\ncol3:", cols3)