import xlrd
import os
from xlutils.copy import copy
# from xlwt import Style


def writeExcel(row, col, str):
    file = './result.xls'
    rb = xlrd.open_workbook(file, formatting_info=True)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    ws.write(row, col, str)
    wb.save(file)


writeExcel(31, 0, 66666)
