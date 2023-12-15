import glm

FOV = 50
NEAR = 0.1
FAR = 100

class Camera:
      def __init__(self,app):
          self.app = app
          self.aspect_ratio = app.WIN_SIZE[0]/app.WIN_SIZE[1]
          self.position = glm.vec3(2,3,3)
          self.up = glm.vec3(0,1,0)
          #view matrix
          self.m_view = self.get_view_matrix()
          #projection matrix
          self.m_proj = self.get_projection_matrix()

      def get_view_matrix(self):
          return glm.lookAt(self.position,glm.vec3(0),self.up)

      def get_projection_matrix(self):
          return glm.perspective(glm.radians(FOV), self.aspect_ratio, NEAR, FAR)

      def screen_to_world(self, screen_coordinates):
            # Convert screen coordinates to NDC coordinates
            ndc_coordinates = glm.vec2(
                2.0 * screen_coordinates[0] / self.app.WIN_SIZE[0] - 1.0,
                1.0 - 2.0 * screen_coordinates[1] / self.app.WIN_SIZE[1]
            )

            # Create a 4D homogeneous clip-space coordinate
            clip_coordinates = glm.vec4(ndc_coordinates.x, ndc_coordinates.y, -1.0, 1.0)

            # Get the inverse of the combined view-projection matrix
            inverse_mvp = glm.inverse(self.m_proj * self.m_view)

            # Transform clip coordinates to world coordinates
            world_coordinates = inverse_mvp * clip_coordinates
            world_coordinates /= world_coordinates.w

            return glm.vec3(world_coordinates)

      
      def world_to_screen(self, world_position):
        # Get the model-view-projection matrix
            mvp_matrix = self.m_proj * self.m_view

            # Transform the world position to clip coordinates
            clip_coordinates = mvp_matrix * glm.vec4(world_position, 1.0)

            # Homogeneous divide to get normalized device coordinates (NDC)
            ndc_coordinates = clip_coordinates.xyz / clip_coordinates.w

            # Convert NDC to screen coordinates
            screen_x = (ndc_coordinates.x + 1.0) * 0.5 * self.app.WIN_SIZE[0]
            screen_y = (1.0 - ndc_coordinates.y) * 0.5 * self.app.WIN_SIZE[1]

            return glm.vec2(screen_x, screen_y)

      def set_position(self, new_position):
            self.position = new_position
            self.m_view = self.get_view_matrix()