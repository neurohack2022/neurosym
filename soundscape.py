import time
from agent import Conducter
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import threading
from tkinter import *
from functools import partial


class soundscape:
   
    
    def main(qq):

        # Optional argument, if not specified will search path.
        
        driver = webdriver.Chrome(executable_path='chromedriver')
        driver.get('https://mynoise.net/NoiseMachines/windSeaRainNoiseGenerator.php')

        time.sleep(5)  # Let the user actually see something!

        try:

            play = driver.find_element("class name", "contextPlay")

            play.click()
        except:
            print("Element probably wasn't found")

        time.sleep(3)  # Let the user actually see something!

        for i in range(10):

            slider=driver.find_element("id", "s"+str(i))
            ac = ActionChains(driver)

            ac.move_to_element(slider).move_by_offset(0,0).click().perform()

        win = Tk()
        win.geometry("1000x1000")
        button1 =  Button(win, text="Train", command=partial(soundscape.train,driver,qq),height = 20,width = 40)
        #put on screen
        button1.pack()
        button2 =  Button(win, text="Run", command=partial(soundscape.run,driver,qq),height = 20,width = 40)
        #put on screen
        button2.pack()
        win.mainloop()
        
        

    def train(driver,qq):
        agent = Conducter()
        agent.train(driver,qq)
        agent.plot_learning()
    
    def run(driver,qq):
        agent = Conducter()
        agent.run(driver,qq)


if __name__ == '__main__':
    soundscape().main()
