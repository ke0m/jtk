import random
import threading
import time


from java.lang import System, Runnable
from javax.swing import (BoxLayout, ImageIcon, JButton, JFrame, JPanel,
        JPasswordField, JLabel, JTextArea, JTextField, JScrollPane,
        SwingConstants, WindowConstants, Timer, SwingUtilities)
from java.awt import Component, GridLayout, BorderLayout, Color, FlowLayout, Font
from javax.imageio import ImageIO 
from java import io
from   java.util      import Calendar

from java.lang import Runnable

from edu.mines.jtk.util.ArgsParser import *

import java_smoother
import jocl_smoother

from java_smoother import *
from jocl_smoother import *



class Demo(JFrame, Runnable):

    def __init__(self):
        super(Demo, self).__init__()

        self.initUI()

    def initUI(self):
       
        self.panel = JPanel(size=(50,50))
        

        self.panel.setLayout(FlowLayout( ))
        self.panel.setToolTipText("GPU Demo")

        self.textfield1 = JTextField('Smoothing Parameter',15)        
        self.panel.add(self.textfield1)
      
        
        joclButton = JButton("JOCL",actionPerformed=self.onJocl)
        joclButton.setBounds(100, 500, 100, 30)
        joclButton.setToolTipText("JOCL Button")
        self.panel.add(joclButton)
        
        
        javaButton = JButton("Java",actionPerformed=self.onJava)
        javaButton.setBounds(100, 500, 100, 30)
        javaButton.setToolTipText("Java Button")
        self.panel.add(javaButton)

        qButton = JButton("Quit", actionPerformed=self.onQuit)
        qButton.setBounds(200, 500, 80, 30)
        qButton.setToolTipText("Quit Button")
        self.panel.add(qButton)
		# I should set this to an environment variable
        newImage = ImageIO.read(io.File("/Users/joe/dev/jtk/data/input.png"))
        resizedImage =  newImage.getScaledInstance(600, 600,10)
        newIcon = ImageIcon(resizedImage)
        label1 = JLabel("Input Image",newIcon, JLabel.CENTER)

        label1.setVerticalTextPosition(JLabel.TOP)
        label1.setHorizontalTextPosition(JLabel.RIGHT)
        label1.setSize(10,10)
        label1.setBackground(Color.orange)
        self.panel.add(label1)
        
        self.getContentPane().add(self.panel)
        
        self.clockLabel = JLabel()
        self.clockLabel.setSize(1,1)
        self.clockLabel.setBackground(Color.orange)
        
        self.clockLabel.setVerticalTextPosition(JLabel.BOTTOM)
        self.clockLabel.setHorizontalTextPosition(JLabel.LEFT)
        
        myClockFont = Font("Serif", Font.PLAIN, 50)
        self.clockLabel.setFont(myClockFont)
        
        
        self.panel.add(self.clockLabel)
        
        self.setTitle("GPU Demo")
        self.setSize(1200, 600)
        self.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
        self.setLocationRelativeTo(None)
        self.setVisible(True)

    def onQuit(self, e): System.exit(0)
    
    def onJocl(self, e): 
     self.clockLabel.setText('running')
     self.started = Calendar.getInstance().getTimeInMillis();
     #print self.textfield1.getText()  
     #time.sleep(5)
     iters = toInt(self.textfield1.getText())
     jocl_smoother(iters)
     elapsed = Calendar.getInstance().getTimeInMillis() - self.started;
     self.clockLabel.setText( 'JOCL Elapsed: %.2f seconds' % ( float( elapsed ) / 1000.0 ) )
    
    def onJava(self, e): 
     self.clockLabel.setText('running')
     self.started = Calendar.getInstance().getTimeInMillis();
     #print self.textfield1.getText()  
     #time.sleep(5)
     iters = toInt(self.textfield1.getText())
     java_smoother(iters)
     elapsed = Calendar.getInstance().getTimeInMillis() - self.started;
     self.clockLabel.setText( 'Java Elapsed: %.2f seconds' % ( float( elapsed ) / 1000.0 ) )
     
    
if ( __name__ == '__main__' ) or ( __name__ == 'main' ) :
    SwingUtilities.invokeLater( Demo() )
    raw_input();
    sys.exit();
else :
  print 'Error - script must be executed, not imported.\n';
  print 'Usage: jython %s.py\n' % __name__;
  print '   or: wsadmin -conntype none -f %s.py' % __name__;
