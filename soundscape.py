import time
from agent import Conducter
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import threading


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



        agent = Conducter()
        agent.train(driver,qq)
        agent.plot_learning()

if __name__ == '__main__':
    soundscape().main()
