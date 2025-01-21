
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

@dataclass
class Point:
    x: float
    y: float
    
    def to_pixel_coords(self, size: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates"""
        return (int(self.x * size), int(self.y * size))

class Brush:
    def __init__(self, control_points: List[Point], grayscale: float):
        """
        Initialize a brush with 6 control points and grayscale value.
        
        Args:
            control_points: List of 6 Point objects in normalized coordinates (0-1)
            grayscale: Float value between 0-1 representing brush darkness
        """
        if len(control_points) != 6:
            raise ValueError("Brush must have exactly 6 control points")
        if not 0 <= grayscale <= 1:
            raise ValueError("Grayscale value must be between 0 and 1")
            
        self.control_points = control_points
        self.grayscale = grayscale
        
    def get_bezier_curves(self) -> List[Tuple[Point, Point, Point, Point]]:
        """Split control points into two cubic Bezier curves"""
        return [
            # First curve: points 0-3
            (self.control_points[0], self.control_points[1], 
             self.control_points[2], self.control_points[3]),
            # Second curve: points 3-6
            (self.control_points[3], self.control_points[4], 
             self.control_points[5], self.control_points[0])
        ]
    
    def get_curve_points(self, size: int, steps: int = 500) -> List[Tuple[int, int]]:
        """Generate points along both Bezier curves"""
        curves = self.get_bezier_curves()
        all_points = []
        
        for p0, p1, p2, p3 in curves:
            for t in np.linspace(0, 1, steps):
                # Cubic Bezier formula
                x = (1-t)**3 * p0.x + 3*(1-t)**2*t * p1.x + \
                    3*(1-t)*t**2 * p2.x + t**3 * p3.x
                y = (1-t)**3 * p0.y + 3*(1-t)**2*t * p1.y + \
                    3*(1-t)*t**2 * p2.y + t**3 * p3.y
                all_points.append((int(x * size), int(y * size)))
                
        return all_points
    
    def create_brush_layer(self, size: int, scale: int = 2) -> Image.Image:
        """Create a brush stroke layer with smooth edges"""
        temp_size = size * scale
        layer = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        
        # Get points for the brush stroke
        points = self.get_curve_points(temp_size)
        
        # Calculate color based on grayscale value
        gray = int(255 * (1 - self.grayscale))
        color = (gray, gray, gray, 255)
        
        # Draw main shape
        draw.polygon(points, fill=color)
        
        # Add smooth edges
        for i in range(scale):
            alpha = int(255 * (1 - (i / scale)))
            edge_color = color[:-1] + (alpha,)
            draw.line(points + [points[0]], fill=edge_color, width=i+1)
            
        # Resize to final size
        return layer.resize((size, size), Image.LANCZOS)
    
    def draw(self, size: int, scale: int = 2) -> Image.Image:
        """Create a complete brush stroke image"""
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        brush_layer = self.create_brush_layer(size, scale)
        return Image.alpha_composite(image.convert('RGBA'), brush_layer)
points = [
    Point(0.5, 0.75),
    Point(0.625, 0.75),
    Point(0.75, 0.625),
    Point(0.75, 0.5),
    Point(0.75+0.01, 0.625),
    Point(0.625, 0.75+0.01),

    
]

brush = Brush(points, grayscale=1)

# Generate and save brush stroke image
image = brush.draw(size=512, scale=2)
image.show()


