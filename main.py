import pygame as pg
import moderngl as mgl
import sys
from model import *
from camera import Camera
from pygame.locals import *
import pyautogui
import tkinter as tk
from tkinter import ttk
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import ctypes


class GraphicsEngine:
    def __init__(self,win_size=(200,200),window_position=(400, 200)):
       #initialize pygmae modules
       pg.init()
       #window size
       self.WIN_SIZE = win_size
      

       # set opengl attr
       pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION,3)
       pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION,3)
       pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,pg.GL_CONTEXT_PROFILE_CORE)

       os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_position[0]},{window_position[1]}"

       #create opengl context
       pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
      
       

       #detect and use opengl context 
       self.ctx = mgl.create_context()
       #self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
       # create an object to keep track time
       self.clock = pg.time.Clock()
       #Camera
       self.camera = Camera(self)
       #scene
       self.scene = Quad(self)
       
       self.points = []  # Store drawn points
       self.mouse_pos = (0, 0)
       self.init_gui()
        
    def init_gui(self):
        # Create a simple tkinter window
        self.gui_root = tk.Tk()
        self.gui_root.title("3D Image Generator")

        self.rotate_button = ttk.Button(self.gui_root, text="Rotate Quad", command=self.rotate_quad)
        self.rotate_button.pack(pady=10)

         # Add a button to the GUI
        self.draw_button = ttk.Button(self.gui_root, text="Draw Mode", command=self.set_draw)
        self.draw_button.pack(pady=10)

        # Add a button to the GUI
        self.generate_button = ttk.Button(self.gui_root, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=10)

        # Add a button to clear the drawing
        self.evaluate_button = ttk.Button(self.gui_root, text="Evaluate", command=self.evaluate)
        self.evaluate_button.pack(pady=10)

        # Add a button to clear the drawing
        self.clear_button = ttk.Button(self.gui_root, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack(pady=10)
        
        window_position = (900, 200)  # Adjust coordinates as needed
        self.gui_root.geometry(f"+{window_position[0]}+{window_position[1]}")

    def rotate_quad(self):
        # Rotate the quad by a certain angle (e.g., 45 degrees)
        self.scene.rotation_angle += 45.0
        # Re-render the scene with the updated quad rotation
        self.render()

    def set_draw(self):
        position = glm.vec3(0,0,3)
        self.camera.set_position(position)
        # Re-render the scene with the updated quad 
        self.render()

    def get_window_position(self):
        # Get the window handle
        hwnd = pg.display.get_wm_info()["window"]

        # Get the client area of the window
        client_rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))

        # Convert the client area coordinates to screen coordinates
        point = ctypes.wintypes.POINT(client_rect.left, client_rect.top)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))

        return point.x, point.y

    def generate_image(self):
        

        window_x, window_y = self.get_window_position()

        #print(window_x,window_y)
        
        filename="output.png"
        screenshot = pyautogui.screenshot(region=(window_x,window_y, self.WIN_SIZE[0], self.WIN_SIZE[1]))

        screenshot_rgba = screenshot.convert("RGBA")

        # Convert the screenshot to a Pygame surface
        img = pg.image.fromstring(screenshot_rgba.tobytes(), screenshot.size, "RGBA")

        # Save the image (optional)
        pg.image.save(img, "output.png")
        
        #print("Screenshot saved as screenshot.png")

    def clear_drawing(self):
        self.points = []  # Clear the list of drawn points

    def evaluate(self):
        model = keras.models.load_model('trained-model.h5')
        IMG_SIZE=28

        img = cv2.imread('output.png')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)

        newimg = tf.keras.utils.normalize(resized, axis = 1)

        newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        predictions = model.predict(newimg)
        print(np.argmax(predictions[0]))
        self.plot_bar_chart(predictions)

    def plot_bar_chart(self, predictions):
        classes = list(range(10))  # Adjust the number of classes based on your model
        

        plt.bar(classes, predictions[0], color='blue')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Model Predictions')
        plt.show()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
               self.scene.destroy()
               pg.quit()
               sys.exit()
            elif event.type == pg.MOUSEMOTION:
                if pg.mouse.get_pressed()[0]:  # Left mouse button is pressed
                    current_pos = event.pos
                    if self.points:  # Check if there's a previous point
                        prev_pos = self.points[-1]
                        # Add the line segment between current and previous points
                        self.points.extend(self.get_line_points(prev_pos, current_pos))
                    else:
                        self.points.append(current_pos)

            

    def get_line_points(self, start, end):
        # Bresenham's Line Algorithm to get all points on the line
        points = []
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points
               
    
    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # scene render
        self.scene.render()
        # Draw previously clicked points
        if self.points:
            self.ctx.enable(mgl.BLEND)
            point_shader = self.ctx.program(
                vertex_shader='''#version 330
                    in vec3 in_vert;
                    out float distanceToCenter;  // Output distance to center

                    uniform mat4 mvp;

                    void main() {
                        gl_Position = mvp * vec4(in_vert, 1.0);
                        gl_PointSize = 100.0;  // Adjust the point size as needed

                        // Calculate distance to the center in the vertex shader
                        distanceToCenter = length(in_vert.xy);
                    }

                ''',
                fragment_shader='''#version 330
                    out vec4 fragColor;

                    void main() {
                        fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Set color to red
                    }

                '''
            )

            # Convert mouse positions to NDC
            mouse_positions_ndc = [
                glm.vec2((2.0 * pos[0]) / self.WIN_SIZE[0] - 1.0, 1.0 - (2.0 * pos[1]) / self.WIN_SIZE[1])
                for pos in self.points
            ]

            # Convert NDC to world space
            world_positions = [
                glm.vec3(glm.inverse(glm.mat4()) * glm.vec4(mouse_ndc, 0.0, 1.0))
                for mouse_ndc in mouse_positions_ndc
            ]

            vertices = np.array(world_positions, dtype=np.float32)
            vertex_data = vertices.tobytes()

            vbo = self.ctx.buffer(vertex_data)
            vao = self.ctx.simple_vertex_array(point_shader, vbo, 'in_vert')
            point_shader['mvp'].write(glm.mat4())  # Identity matrix for 2D rendering
            vao.render(mode=mgl.LINES)
            self.ctx.disable(mgl.BLEND)


   
        # swap buffers
        pg.display.flip()
        
    def run(self):
        while True:
            self.check_events()
            self.gui_root.update()
            self.render()
            self.delta_time = self.clock.tick(60)
            
    
    
if __name__ == '__main__':
   app = GraphicsEngine()
   app.run()
            
            