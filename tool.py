#QMenuBar/QMenu/QAction的使用(菜单栏）
from PyQt5.QtWidgets import   QMenuBar,QMenu,QAction,QLineEdit,QStyle,QFormLayout,   QVBoxLayout,QWidget,QApplication ,QHBoxLayout, QPushButton,QMainWindow,QGridLayout,QLabel
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QIcon,QPixmap,QFont
from PyQt5.QtCore import  QDate
from PyQt5.QtWidgets import QFileDialog,QTextEdit
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import os 

import sys

global rectStart_x#矩形左上角x坐标
global rectStart_y#矩形左上角y坐标
global rectEnd_x#矩形右下角x坐标
global rectEnd_y#矩形右下角y坐标
global pointX#点x坐标
global pointY#点y坐标
global data1,data2,data3,data4,data5#文本框
global P_i#选中点的坐标

global circleX#检验圆的点X列表
global circleY#检验圆的点Y列表
global cFlag

rectStart_x = []
rectStart_y = []
rectEnd_x = []
rectEnd_y = []
pointX = []
pointY = []
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
P_i = 0

circleX = []
circleY = []
cFlag = []
cFlag.append(False)


class WindowClass(QMainWindow):
	
	i = 0#文件夹的第i张图片
	path = ""
	fileList = ""
	txtList = ""
	
	def __init__(self,parent=None):
		super(WindowClass, self).__init__(parent)

		self.menubar = self.menuBar()#获取窗体的菜单栏
		self.resize(1500,900)

		self.file = self.menubar.addMenu("系统菜单")
		self.file.addAction("打开文件")

		self.save = QAction("保存",self)
		self.save.setShortcut("Ctrl+S")#设置快捷键
		self.file.addAction(self.save)
		
		self.clear = QAction("清空",self)
		self.file.addAction(self.clear)

		#self.edit = self.file.addMenu("编辑")
		#self.edit.addAction("copy")#Edit下这是copy子项
		#self.edit.addAction("paste")#Edit下设置paste子项

		#self.quit = QAction("Quit",self)#注意如果改为：self.file.addMenu("Quit") 则表示该菜单下必须柚子菜单项；会有>箭头
		#self.file.addAction(self.quit)
		self.file.triggered[QAction].connect(self.openFile)
		
		self.btn = QPushButton(self)
		self.btn.clicked.connect(self.nextPhoto)
		self.btn.setText("下一张")
		self.btn.resize(100,30)
		self.btn.move(200,850)
		
		self.btn_2 = QPushButton(self)
		self.btn_2.clicked.connect(self.prePhoto)
		self.btn_2.setText("上一张")
		self.btn_2.resize(100,30)
		self.btn_2.move(350,850)

		self.btn_3 = QPushButton(self)
		self.btn_3.clicked.connect(MyLabel.conculate1)
		self.btn_3.setText("计算矩阵")
		self.btn_3.resize(100,30)
		self.btn_3.move(50,850)
		
		self.setWindowTitle("Menu Demo")
		'''
		lb = QLabel(self)
		lb.setGeometry = (100,500,500,300)
		lb.setStyleSheet("border:2px solid red")
		'''
	def openFile(self,qaction):
		if qaction.text() == "打开文件":
			self.path = QFileDialog.getExistingDirectory(None,"选取文件夹","./")
			#i = 0
			print(self.path)
			self.fileList = os.listdir(self.path)
			global lenth
			lenth = len(self.fileList)
			print(self.i)
			print(lenth)
			
			#init i
			self.i=1
			while True:
				if self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'jpg' or self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'png' or self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'gif':
					print(self.i)
					break
				self.i += 1
			
			print(self.fileList)
			print(self.path + '/' + self.fileList[self.i])
			image = QPixmap(self.path + '/' + self.fileList[self.i])
			#图片标签
			global lbImage
			lbImage = MyLabel(self)
			lbImage.setFocusPolicy(11)
			lbImage.resize(1200,800)
			lbImage.move(150,20)
			lbImage.setScaledContents(True)
			lbImage.show()
			#lb.setStyleSheet("border:2px solid red")
			lbImage.setPixmap(image)
			#文本
			global Text_1,Text_2,Text_3,Text_4,Text_5
			Text_1 = QTextEdit(self)
			Text_1.resize(100,30)
			Text_1.move(500,850)
			Text_1.show()

			Text_2 = QTextEdit(self)
			Text_2.resize(100,30)
			Text_2.move(650,850)
			Text_2.show()
		
			Text_3 = QTextEdit(self)
			Text_3.resize(100,30)
			Text_3.move(800,850)
			Text_3.show()

			Text_4 = QTextEdit(self)
			Text_4.resize(100,30)
			Text_4.move(950,850)
			Text_4.show()

			Text_5 = QTextEdit(self)
			Text_5.resize(100,30)
			Text_5.move(1100,850)
			Text_5.show()

			self.openTxt()
			
		elif qaction.text() == "保存":
			if data1 != []:
				del data1[0]
			data1.append(Text_1.toPlainText())
			if data2 != []:
				del data2[0]
			data2.append(Text_2.toPlainText())
			if data3 != []:
				del data3[0]
			data3.append(Text_3.toPlainText())
			if data4 != []:
				del data4[0]
			data4.append(Text_4.toPlainText())
			if data5 != []:
				del data5[0]
			data5.append(Text_5.toPlainText())

			tName = self.fileList[self.i][:len(self.fileList[self.i]) - 3] + 'txt'
			f = open(self.path + '/' + "annotation" + '/' + tName,'w')
			f.write(str(rectStart_x[0]) + '\n' + str(rectStart_y[0]) + '\n' + str(rectEnd_x[0]) + '\n' + str(rectEnd_y[0]) + '\n')

			n = 0
			plenth = len(pointX)
			while n < plenth:
				f.write(str(pointX[n]) + '\t')
				n += 1
			f.write('\n')
			
			n = 0
			while n < plenth:
				f.write(str(pointY[n]) + '\t')
				n += 1
			f.write('\n')

			f.write(str(data1[0]) + '\n')
			f.write(str(data2[0]) + '\n')
			f.write(str(data3[0]) + '\n')
			f.write(str(data4[0]) + '\n')
			f.write(str(data5[0]))

			f.close()
			print(str(self.i)+' image ok')
			
		elif qaction.text() == "清空":
			if rectStart_x != []:
				del rectStart_x[0]
				if rectStart_y != []:
					del rectStart_y[0]
				if rectEnd_x != []:
					del rectEnd_x[0]
				if rectEnd_y != []:
					del rectEnd_y[0]
				if data1 != []:
					del data1[0]
					Text_1.setPlainText('')
				if data2 != []:
					del data2[0]
					Text_2.setPlainText('')
				if data3 != []:
					del data3[0]
					Text_3.setPlainText('')
				if data4 != []:
					del data4[0]
					Text_4.setPlainText('')
				if data5 != []:
					del data5[0]
					Text_5.setPlainText('')
				if pointX != []:
					del pointX[:]
				if pointY != []:
					del pointY[:]
			
		
	def nextPhoto(self):
		while True:
			self.i += 1	
			if self.i >= lenth:
				print("这是最后一张了!")
				self.i -= 1
				return
			if self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'jpg' or self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'png' or self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'gif':
				break

		print(self.path + '/' + self.fileList[self.i])
		image = QPixmap(self.path + '/' + self.fileList[self.i])
		#lbImage = QLabel(self)
		#from openFile import lbImage
		lbImage.resize(1200,800)
		lbImage.move(150,20)
		lbImage.setScaledContents(True)
		lbImage.show()
		lbImage.setPixmap(image)
		self.openTxt()
		#lbImage.setStyleSheet("border:2px solid red")
		
	def prePhoto(self):
		while True:
			self.i -= 1
			if self.i < 0:
				print("这是第一张了!")
				self.i += 1
				return
			if self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'jpg' or self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'png' or self.fileList[self.i][len(self.fileList[self.i]) - 3:] == 'gif':
				break
			

		print(self.path + '/' + self.fileList[self.i])
		image = QPixmap(self.path + '/' + self.fileList[self.i])
		#lbImage = QLabel(self)
		#from openFile import lbImage
		lbImage.resize(1200,800)
		lbImage.move(150,20)
		lbImage.setScaledContents(True)
		lbImage.show()
		lbImage.setPixmap(image)
		self.openTxt()
		#lbImage.setStyleSheet("border:2px solid red")


	#读取图片的文件
	def openTxt(self):
		self.txtList = os.listdir(self.path + '/' + "annotation")
		lenth = len(self.txtList)
		MyLabel.delete(lbImage)
		for name in self.txtList:
			#print(name)
			if name == self.fileList[self.i][:len(self.fileList[self.i]) - 3] + 'txt':
				f = open(self.path + '/' + "annotation" + '/' + name,'r')
				if rectStart_x != []:
					del rectStart_x[0]
					del rectStart_y[0]
					del rectEnd_x[0]
					del rectEnd_y[0]
					del data1[0]
					del data2[0]
					del data3[0]
					del data4[0]
					del data5[0]
					del pointX[:]
					del pointY[:]
					
				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					rectStart_x.append(int(temp))
					
				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					rectStart_y.append(int(temp))
					
				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					rectEnd_x.append(int(temp))
					
				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					rectEnd_y.append(int(temp))
					
				temp = f.readline()
				if temp != '\n':
					i = 0
					j = 0

					while temp[i] != '\n':
						if temp[i] == '\t':
							pointX.append(int(temp[j:i]))
							j = i + 1
						i += 1
						
				temp = f.readline()
				if temp != '\n':
					i = 0
					j = 0
					del pointY[:]
					while temp[i] != '\n':
						if temp[i] == '\t':
							pointY.append(int(temp[j:i]))
							j = i + 1
						i += 1
						
				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					data1.append(temp.strip('\n'))
					if data1 != []:
						Text_1.setPlainText(str(data1[0]))

				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					data2.append(temp.strip('\n'))
					if data2 != []:
						Text_2.setPlainText(str(data2[0]))

				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					data3.append(temp.strip('\n'))
					if data3 != []:
						Text_3.setPlainText(str(data3[0]))

				temp = f.readline()
				if temp != '\n':
					temp.strip('\n')
					data4.append(temp.strip('\n'))
					if data4 != []:
						Text_4.setPlainText(str(data4[0]))

				temp = f.readline()
				if temp != '':
					temp.strip('\n')
					data5.append(temp.strip('\n'))
					if data5 != []:
						Text_5.setPlainText(str(data5[0]))

				f.close()
				break
			else:
				if rectStart_x != []:
					del rectStart_x[0]
				if rectStart_y != []:
					del rectStart_y[0]
				if rectEnd_x != []:
					del rectEnd_x[0]
				if rectEnd_y != []:
					del rectEnd_y[0]
				if data1 != []:
					del data1[0]
					Text_1.setPlainText('')
				if data2 != []:
					del data2[0]
					Text_2.setPlainText('')
				if data3 != []:
					del data3[0]
					Text_3.setPlainText('')
				if data4 != []:
					del data4[0]
					Text_4.setPlainText('')
				if data5 != []:
					del data5[0]
					Text_5.setPlainText('')
				if pointX != []:
					del pointX[:]
				if pointY != []:
					del pointY[:]




class MyLabel(QLabel):
	x0 = 0
	y0 = 0
	x1 = 0
	y1 = 0
	pX = 0
	pY = 0
	flag = 0
	#鼠标点击事件
	def mousePressEvent(self,event):
		if event.button() == 2:
			self.flag = 2
			self.x0 = event.x()
			self.y0 = event.y()
			self.x1 = event.x()
			self.y1 = event.y()
		elif event.button() == 1:
			self.flag = 1
			self.pX = event.x()
			self.pY = event.y()
	#鼠标释放事件
	def mouseReleaseEvent(self,event):
		if self.flag == 1:
			re = False
			#判断是否存在该点
			i = 0
			global P_i
			P_i = 0
			lent = len(pointX)
			while i < lent:
				if abs(self.pX - pointX[i]) <= 5 and abs(self.pY - pointY[i]) <= 5:
					P_i = i
					re = True
				i += 1
			if re == False:
				pointX.append(self.pX)
				pointY.append(self.pY)
				P_i = len(pointX) - 1
		elif self.flag == 2:
			if rectStart_x != []:
				del rectStart_x[0]
				del rectStart_y[0]
				del rectEnd_x[0]
				del rectEnd_y[0]
			if self.x0 > self.x1:
				t = self.x0
				self.x0 = self.x1
				self.x1 = t
			if self.y0 > self.y1:
				t = self.y0
				self.y0 = self.y1
				self.y1 = t
			
			rectStart_x.append(self.x0)
			rectStart_y.append(self.y0)
			rectEnd_x.append(self.x1)
			rectEnd_y.append(self.y1)
		self.flag = 0

	#鼠标移动事件
	def mouseMoveEvent(self,event):
		if self.flag == 2:
			self.x1 = event.x()
			self.y1 = event.y()
		self.update()

	#按下键盘事件
	def keyPressEvent(self, QKeyEvent):
		if QKeyEvent.key() == Qt.Key_Left:
			pointX[P_i] -= 1
			MyLabel.conculate1()
		if QKeyEvent.key() == Qt.Key_Right:
			pointX[P_i] += 1
			MyLabel.conculate1()
		if QKeyEvent.key() == Qt.Key_Up:
			pointY[P_i] -= 1
			MyLabel.conculate1()
		if QKeyEvent.key() == Qt.Key_Down:
			pointY[P_i] += 1
			MyLabel.conculate1()
		if QKeyEvent.key() == Qt.Key_Delete:
			del pointX[P_i]
			del pointY[P_i]
			MyLabel.conculate1()


	#绘制事件
	def paintEvent(self,event):
		super().paintEvent(event)
		painter = QPainter(self)
		if rectStart_x != []:
			rect = QRect(rectStart_x[0], rectStart_y[0], abs(rectStart_x[0] - rectEnd_x[0]), abs(rectEnd_y[0] - rectStart_y[0]))
			painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
			painter.drawRect(rect)
			self.update()
		if pointX != []:
			i = 0
			lent = len(pointX)
			while i < lent:
				if i >= 0 and i < 4:
					painter.setPen(QPen(Qt.red,5,Qt.SolidLine))
				elif i >= 4:
					if i % 4 == 0:
						painter.setPen(QPen(Qt.yellow,5,Qt.SolidLine))
					elif i % 4 == 1:
						painter.setPen(QPen(Qt.green,5,Qt.SolidLine))
					elif i % 4 == 2:
						painter.setPen(QPen(Qt.white,5,Qt.SolidLine))
					elif i % 4 == 3:
						painter.setPen(QPen(Qt.blue,5,Qt.SolidLine))
				painter.drawPoint(pointX[i],pointY[i])
				i += 1
				self.update()
		if self.x0 < self.x1 and self.y0 < self.y1:
			rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
		elif self.x0 < self.x1 and self.y0 > self.y1:
			rect = QRect(self.x0, self.y1, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
		elif self.x0 > self.x1 and self.y0 < self.y1:
			rect = QRect(self.x1, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
		elif self.x0 > self.x1 and self.y0 > self.y1:
			rect = QRect(self.x1, self.y1, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
		else:
			rect = QRect(0,0,0,0)
		painter.setPen(QPen(Qt.white,2,Qt.SolidLine))
		painter.drawRect(rect)

		if cFlag[0] == True:
			if len(pointX) >= 8:
				i = 0
				while i < len(circleX):
					painter.setPen(QPen(Qt.yellow,2,Qt.SolidLine))
					painter.drawPoint(circleX[i],circleY[i])
					i += 1


	def delete(self):#切换图片后清空临时矩形
		self.x0 = self.x1 = self.y0 = self.y1 = 0
		cFlag[0] = False
		del circleX[:]
		del circleY[:]
	
	
	def conculate1():
		if circleX != []:
			del circleX[:]
			del circleY[:]
		i = 1
		ci = 0
		while i <= 4:
			if len(pointX) >= 8:
				cFlag[0] = True
				np.a = [
					[pointX[4*i],pointY[4*i],1,0,0,0,0,0],
					[0,0,0,pointX[4*i],pointY[4*i],1,pointX[4*i],pointY[4*i]],
					[pointX[1 + 4*i],pointY[1 + 4*i],1,0,0,0,-np.sin(108 * np.pi / 180) * pointX[1 + 4*i],-np.sin(108 * np.pi / 180) * pointY[1 + 4*i]],
					[0,0,0,pointX[1 + 4*i],pointY[1 + 4*i],1,np.cos(108 * np.pi / 180) * pointX[1 + 4*i],np.cos(108 * np.pi / 180) * pointY[1 + 4*i]],
					[pointX[2 + 4*i],pointY[2 + 4*i],1,0,0,0,0,0],
					[0,0,0,pointX[2 + 4*i],pointY[2 + 4*i],1,-pointX[2 + 4*i],-pointY[2 + 4*i]],
					[pointX[3 + 4*i],pointY[3 + 4*i],1,0,0,0,-np.sin(252 * np.pi / 180) * pointX[3 + 4*i],-np.sin(252 * np.pi / 180) * pointY[3 + 4*i]],
					[0,0,0,pointX[3 + 4*i],pointY[3 + 4*i],1,np.cos(252 * np.pi / 180) * pointX[3 + 4*i], np.cos(252 * np.pi / 180) * pointY[3 + 4*i]],
					]
				np.b = [
					[0],
					[-1],
					[np.sin(108 * np.pi / 180)],
					[-np.cos(108 * np.pi / 180)],
					[0],
					[1],
					[np.sin(252 * np.pi / 180)],
					[-np.cos(252 * np.pi / 180)]
					]
				np.c = np.linalg.solve(np.a,np.b)
				np.H = [
					[float(np.c[0]),float(np.c[1]),float(np.c[2])],
					[float(np.c[3]),float(np.c[4]),float(np.c[5])],
					[float(np.c[6]),float(np.c[7]),1]
					]
				np.re_H = np.linalg.inv(np.H)
				angle = 90
				while angle < 450:
					circleX.append(np.cos(angle * np.pi / 180))
					circleY.append(-np.sin(angle * np.pi / 180))
					
					angle += 36
				clen = len(circleX)
				while ci < clen:
					np.c1 = [[circleX[ci]],[circleY[ci]],[1]]
					np.c2 = np.matmul(np.re_H,np.c1)
					circleX[ci] = float(np.c2[0]) / float(np.c2[2])
					circleY[ci] = float(np.c2[1]) / float(np.c2[2])
					ci += 1
				if len(pointX) >= 8 + 4 * i:
					i += 1
				else:
					break
					
		
		

if __name__ == "__main__":
	app = QApplication(sys.argv)
	win = WindowClass()
	win.show()
	sys.exit(app.exec_())
