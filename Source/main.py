# Code stolen from, I mean inspired by CodeBullet, as per usual
#import multiprocessing
#import statistics
import time

import pyglet
#import glooey

import maintools
import tictactoe.main
from meeple import Meeple

windowMain = pyglet.window.Window(1200, 800)
windowMPL = pyglet.window.Window(1200,800)

log = maintools.colLogger(name="Main")

from population import Population
from annstatistics import Statistics

from tictactoe.main import tictactoeMain
from xor.main import xorMain
#from mastermind.main import mastermindMain
#from dinorunner.main import dinorunnerMain

statswindow = Statistics()

popcap = 1000
population = Population(popcap, 11, 9) # tictactoe compatible population
#population = Population(popcap, 2, 1) #xor compatible population


genlabel = pyglet.text.Label('23423423',
                             font_name='Times New Roman',
                             font_size=20,
                             x=100, y=750,
                             anchor_x='left', anchor_y='center',
                             color=(0,0,0, 255))

def update(dt):
    global population

    log.logger.info("---------------------------------------")
    log.logger.info("New generation: %d" % population.generation)
    # each update is a generation
    temptime = time.time()
    lasttime = [temptime for i in range(3)]

    maintools.loadingbar.printLoadingbar()

    # Game Section Start

    tictactoe.main.tictactoeMain(population);
    #xorMain(population);

    # bestMeep.brain.printNetwork()

    #if population.bestMeeple.score > 39.999:
    #    log.logger.fatal("Meep solved problem")
    #    final_draw();
    #    pyglet.app.exit();

    # Game Section End


    print("")
    meep1:Meeple = population.pop[0]
    meep2:Meeple = population.pop[1]
    log.logger.info("Meep1: %.1f %.3f" %
                 (meep1.elo.rating, meep1.elo.uncertainty))
    log.logger.info("Meep2: %.1f %.3f" %
                 (meep2.elo.rating, meep2.elo.uncertainty))

    log.logger.info("Games took :%.2fs" % (time.time()-lasttime[0]));

    lasttime[1] = time.time()
    population.naturalSelection()
    log.logger.info("NaS took :%.2fs" % (time.time()-lasttime[1]));

    lasttime[2] = time.time()
    statswindow.update(population.generation,
                       population.genscoresHistor_max,
                       population.genscoresHistor_cur,
                       population.scorehistogHistor,
                       population.speciesScoreHistogram);
    log.logger.info("Stats took :%.2fs" % (time.time()-lasttime[2]));

    log.logger.info("Gen took :%.2fs" % (time.time()-lasttime[0]));





@windowMain.event
def on_draw():
    windowMain.clear()
    pyglet.gl.glClearColor(0.7,0.7,0.7,1)
    population.bestMeeple.brain.drawNetwork(50, 50, 1100, 750)
    #population.pop[len(population.species) + 1].brain.drawNetwork(650, 50, 650, 750)
    genlabel.text = "Generation: "+ str(population.generation)
    genlabel.draw()

@windowMPL.event
def on_draw():
    windowMPL.clear();
    pyglet.gl.glClearColor(0.7,0.7,0.7,1)
    if statswindow.image is not None:
        statswindow.image.blit(-50,-50);

def final_draw():
    population.calculateFitness()
    population.setBestMeeple();

    windowMain.switch_to()
    pyglet.image.get_buffer_manager().get_color_buffer().save('windowMain.png')
    windowMPL.switch_to()
    pyglet.image.get_buffer_manager().get_color_buffer().save('windowMPL.png')


@windowMain.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.BACKSPACE:
        print("Final draw")
        final_draw();
        pyglet.app.exit()
    elif symbol == pyglet.window.key.RETURN:
        final_draw();
        return;


#def getScore(decision:List[float], expected:List[float]):
#    runningSum = 0
#    for i in range(len(decision)):
#        runningSum += 1000/((decision[i] - expected[i])**2+1)
#    return runningSum


if __name__ == "__main__":

    print("Starting Main.py as __main__")

    #in Terminal -> snakeviz source/profiledprofile.prof
    #import cProfile
    #cProfile.run('update(10)', filename="profiledprofile.prof")

    pyglet.clock.schedule_interval_soft(update, 1)
    pyglet.app.run()

    print("Finished Main.py as __main__")