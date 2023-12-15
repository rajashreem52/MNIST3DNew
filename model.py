import numpy as np
import glm
import pygame as pg
import moderngl as mgl

class Quad:
      def __init__(self,app):
          self.app = app
          self.ctx = app.ctx
          self.vbo = self.get_vbo()
          self.shader_program = self.get_shaders_program('default')
          self.vao = self.get_vao()
          self.m_model = self.get_model_matrix()
          self.texture = self.get_texture(path = 'textures/test.png')
          self.rotation_angle = 0.0
          self.on_init()

      def get_texture(self,path):
          texture = pg.image.load(path).convert()
          
          texture.fill('black')
          texture = self.ctx.texture (size = texture.get_size(),components =3,
                                      data = pg.image.tostring(texture,'RGB'))
          
          return texture

      def get_model_matrix(self):
          m_model = glm.mat4()
          return m_model

      def on_init(self):
          self.shader_program['u_texture_0'] = 0
          self.texture.use()
          self.shader_program['m_proj'].write(self.app.camera.m_proj)
          self.shader_program['m_view'].write(self.app.camera.m_view)
          self.shader_program['m_model'].write(self.m_model)

      def render(self):
          self.vao.render()
          model_matrix = glm.mat4(1.0)
          model_matrix = glm.rotate(model_matrix, glm.radians(self.rotation_angle), glm.vec3(0, 0, 1))
        
          # Pass the model matrix to the shader
          self.shader_program['m_model'].write(model_matrix)

          self.shader_program['m_view'].write(self.app.camera.m_view)

          # Draw the quad
          self.vao.render(mode=mgl.TRIANGLE_STRIP)

      def destroy(self):
          self.vbo.release()
          self.shader_program.release()
          self.vao.release()

          
      def get_vao(self):
          vao = self.ctx.vertex_array(self.shader_program,[(self.vbo, '2f 3f','in_texcoord_0','in_position')])
          return vao
          
      def get_vertex_data(self):
          vertices = [(-1,1,1), (-1,-1,1),(1,-1,1),(1,1,1)]
          indices = [(2,1,0),(0,3,2)]
          vertex_data = self.get_data(vertices,indices)
          tex_coord = [(0,0),(1,0),(1,1),(0,1)]
          tex_coord_indices = [(0,1,2),(2,3,0)]
          
          tex_coord_data = self.get_data(tex_coord,tex_coord_indices)
          vertex_data = np.hstack([tex_coord_data,vertex_data])
          return vertex_data

      @staticmethod
      def get_data(vertices,indices):
          data = [vertices[ind] for triangle in indices for ind in triangle]
          return np.array(data,dtype = 'f4')

      def get_vbo(self):
          vertex_data = self.get_vertex_data()
          vbo = self.ctx.buffer(vertex_data)
          return vbo

      def get_shaders_program(self,shader_name):
          with open(f'Shaders/{shader_name}.vert') as file:
              vertex_shader = file.read()
          with open(f'Shaders/{shader_name}.frag') as file:
              fragment_shader = file.read()

          program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
          return program