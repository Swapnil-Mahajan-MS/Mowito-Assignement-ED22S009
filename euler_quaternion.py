#!/usr/bin/env python3
"""
Complete Euler <-> Quaternion Converter with Visualization

Single file solution for:
- Euler angle to Quaternion conversion
- Quaternion to Euler angle conversion
- 3D visualization with matplotlib
- Edge case handling (gimbal lock, normalization, etc.)

Usage:
    # Convert and visualize
    python euler_quaternion.py --mode e2q --roll 45 --pitch 30 --yaw 60 --plot
    
    # Convert only (no plot)
    python euler_quaternion.py --mode e2q --roll 45 --pitch 30 --yaw 60
    
    # Quaternion to Euler with plot
    python euler_quaternion.py --mode q2e --x 0.5 --y 0.5 --z 0.5 --w 0.5 --plot
    
    # Save plot to file
    python euler_quaternion.py --mode e2q --roll 45 --pitch 30 --yaw 60 --plot --save output.png
"""

import math
import argparse
import sys
from typing import Tuple, Optional
from dataclasses import dataclass

# Check for visualization dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/numpy not available. Install with:")
    print("  pip install matplotlib numpy --break-system-packages")
    print("Conversion will still work, but visualization is disabled.\n")


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class EulerAngles:
    """Euler angles in Tait-Bryan ZYX convention (roll, pitch, yaw)."""
    roll: float   # Rotation around X-axis
    pitch: float  # Rotation around Y-axis
    yaw: float    # Rotation around Z-axis
    
    def __str__(self):
        return f"Euler(roll={self.roll:.4f}°, pitch={self.pitch:.4f}°, yaw={self.yaw:.4f}°)"


@dataclass
class Quaternion:
    """Quaternion representation (x, y, z, w)."""
    x: float
    y: float
    z: float
    w: float
    
    def norm(self) -> float:
        """Calculate quaternion norm (magnitude)."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    def __str__(self):
        return f"Quat(x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f}, w={self.w:.6f})"


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def normalize_quaternion(quat: Quaternion) -> Quaternion:
    """
    Normalize quaternion to unit length.
    
    Args:
        quat: Input quaternion
        
    Returns:
        Normalized quaternion
        
    Raises:
        ValueError: If quaternion has near-zero norm
    """
    norm = quat.norm()
    
    if norm < 1e-6:
        raise ValueError(
            f"Cannot normalize quaternion with near-zero norm: {norm:.2e}"
        )
    
    return Quaternion(
        x=quat.x / norm,
        y=quat.y / norm,
        z=quat.z / norm,
        w=quat.w / norm
    )


def euler_to_quaternion(
    euler: EulerAngles,
    degrees: bool = True
) -> Quaternion:
    """
    Convert Euler angles to quaternion.
    
    Convention: Tait-Bryan ZYX (yaw-pitch-roll)
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    Args:
        euler: Euler angles (roll, pitch, yaw)
        degrees: If True, input angles are in degrees
        
    Returns:
        Quaternion representation
    """
    # Convert to radians if needed
    if degrees:
        roll = math.radians(euler.roll)
        pitch = math.radians(euler.pitch)
        yaw = math.radians(euler.yaw)
    else:
        roll = euler.roll
        pitch = euler.pitch
        yaw = euler.yaw
    
    # Compute half angles
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    # ZYX rotation order formula
    quat = Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy
    )
    
    # Normalize to handle numerical errors
    return normalize_quaternion(quat)


def quaternion_to_euler(
    quat: Quaternion,
    degrees: bool = True
) -> EulerAngles:
    """
    Convert quaternion to Euler angles.
    
    Convention: Tait-Bryan ZYX (yaw-pitch-roll)
    
    Args:
        quat: Input quaternion (x, y, z, w)
        degrees: If True, output angles in degrees
        
    Returns:
        Euler angles (roll, pitch, yaw)
        
    Edge cases handled:
    - Gimbal lock (pitch = ±90°)
    - Denormalized quaternions
    """
    # Normalize quaternion to be safe
    quat = normalize_quaternion(quat)
    
    x, y, z, w = quat.x, quat.y, quat.z, quat.w
    
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y-axis rotation) - Handle gimbal lock
    sinp = 2.0 * (w * y - z * x)
    
    # Clamp to [-1, 1] to handle numerical errors
    if sinp >= 1.0:
        pitch = math.pi / 2.0  # 90 degrees (gimbal lock)
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0  # -90 degrees (gimbal lock)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees if requested
    if degrees:
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
    
    return EulerAngles(roll=roll, pitch=pitch, yaw=yaw)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

if VISUALIZATION_AVAILABLE:
    
    class Arrow3D(FancyArrowPatch):
        """Helper class for drawing 3D arrows."""
        
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)
            
        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            
            return np.min(zs)


    def quaternion_to_rotation_matrix(quat: Quaternion) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])


    def draw_coordinate_frame(ax, rotation_matrix=None, origin=(0, 0, 0), 
                             scale=1.0, label_prefix="", alpha=1.0, linewidth=2):
        """Draw a 3D coordinate frame (X=red, Y=green, Z=blue)."""
        
        # Define axis vectors
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        # Apply rotation if provided
        if rotation_matrix is not None:
            axes = axes @ rotation_matrix.T
        
        # Draw arrows
        for i, (axis, color, label) in enumerate(zip(axes, colors, labels)):
            arrow = Arrow3D(
                origin[0], origin[1], origin[2],
                axis[0], axis[1], axis[2],
                mutation_scale=20,
                lw=linewidth,
                arrowstyle="-|>",
                color=color,
                alpha=alpha
            )
            ax.add_artist(arrow)
            
            # Add label
            ax.text(
                origin[0] + axis[0] * 1.15,
                origin[1] + axis[1] * 1.15,
                origin[2] + axis[2] * 1.15,
                f"{label_prefix}{label}",
                color=color,
                fontsize=12,
                fontweight='bold',
                alpha=alpha
            )


    def plot_rotation(euler: EulerAngles, quat: Quaternion, save_path: Optional[str] = None):
        """
        Create a 3D visualization of the rotation.
        
        Args:
            euler: Euler angles
            quat: Quaternion
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Left plot: 3D rotation visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        rot_matrix = quaternion_to_rotation_matrix(quat)
        
        # Draw original frame (faded)
        draw_coordinate_frame(ax1, None, scale=1.0, label_prefix="", alpha=0.3, linewidth=1)
        
        # Draw rotated frame
        draw_coordinate_frame(ax1, rot_matrix, scale=1.0, label_prefix="'", alpha=1.0, linewidth=2.5)
        
        # Set labels and title
        ax1.set_xlabel('X', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax1.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax1.set_title("3D Rotation Visualization", fontsize=14, fontweight='bold', pad=20)
        
        # Set equal aspect ratio
        max_range = 1.5
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_zlim([-max_range, max_range])
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        ax1.view_init(elev=20, azim=45)
        
        # Right plot: Information panel
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        # Create information text
        info_text = f"""
EULER ANGLES (ZYX Convention)
{"="*50}

Roll (X-axis):    {euler.roll:10.4f}°
Pitch (Y-axis):   {euler.pitch:10.4f}°
Yaw (Z-axis):     {euler.yaw:10.4f}°


QUATERNION
{"="*50}

x:    {quat.x:10.6f}
y:    {quat.y:10.6f}
z:    {quat.z:10.6f}
w:    {quat.w:10.6f}

Norm: {quat.norm():10.6f}


ROTATION MATRIX
{"="*50}

{rot_matrix[0,0]:8.4f}  {rot_matrix[0,1]:8.4f}  {rot_matrix[0,2]:8.4f}
{rot_matrix[1,0]:8.4f}  {rot_matrix[1,1]:8.4f}  {rot_matrix[1,2]:8.4f}
{rot_matrix[2,0]:8.4f}  {rot_matrix[2,1]:8.4f}  {rot_matrix[2,2]:8.4f}


COORDINATE FRAME LEGEND
{"="*50}

Original frame (faded):
  🔴 X-axis (red)
  🟢 Y-axis (green)
  🔵 Z-axis (blue)

Rotated frame (bright):
  🔴 X'-axis (red)
  🟢 Y'-axis (green)
  🔵 Z'-axis (blue)


ROTATION ORDER
{"="*50}

R = Rz(yaw) × Ry(pitch) × Rx(roll)
        """
        
        ax2.text(0.05, 0.95, info_text, 
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Main title
        plt.suptitle(f"Euler ↔ Quaternion Conversion\n{euler}\n{quat}", 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def print_conversion_result(mode: str, euler: EulerAngles, quat: Quaternion):
    """Print conversion results in a formatted way."""
    
    print("\n" + "="*70)
    if mode == "e2q":
        print("EULER → QUATERNION CONVERSION")
    else:
        print("QUATERNION → EULER CONVERSION")
    print("="*70)
    
    print("\nInput:")
    if mode == "e2q":
        print(f"  Euler Angles:")
        print(f"    Roll:  {euler.roll:10.4f}°")
        print(f"    Pitch: {euler.pitch:10.4f}°")
        print(f"    Yaw:   {euler.yaw:10.4f}°")
    else:
        print(f"  Quaternion:")
        print(f"    x: {quat.x:10.6f}")
        print(f"    y: {quat.y:10.6f}")
        print(f"    z: {quat.z:10.6f}")
        print(f"    w: {quat.w:10.6f}")
        print(f"    Norm: {quat.norm():10.6f}")
    
    print("\nOutput:")
    if mode == "e2q":
        print(f"  Quaternion:")
        print(f"    x: {quat.x:10.6f}")
        print(f"    y: {quat.y:10.6f}")
        print(f"    z: {quat.z:10.6f}")
        print(f"    w: {quat.w:10.6f}")
        print(f"    Norm: {quat.norm():10.6f}")
    else:
        print(f"  Euler Angles:")
        print(f"    Roll:  {euler.roll:10.4f}°")
        print(f"    Pitch: {euler.pitch:10.4f}°")
        print(f"    Yaw:   {euler.yaw:10.4f}°")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Convert between Euler angles and quaternions with optional 3D visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Euler to Quaternion (with plot)
  %(prog)s --mode e2q --roll 45 --pitch 30 --yaw 60 --plot
  
  # Quaternion to Euler (no plot)
  %(prog)s --mode q2e --x 0.5 --y 0.5 --z 0.5 --w 0.5
  
  # Save visualization to file
  %(prog)s --mode e2q --roll 45 --pitch 30 --yaw 60 --plot --save output.png
  
  # Use radians instead of degrees
  %(prog)s --mode e2q --roll 0.785 --pitch 0.524 --yaw 1.047 --radians
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["e2q", "q2e"],
        required=True,
        help="Conversion mode: e2q (Euler->Quaternion) or q2e (Quaternion->Euler)"
    )
    
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Use radians instead of degrees (default: degrees)"
    )
    
    # Euler angle inputs
    parser.add_argument("--roll", type=float, help="Roll angle (rotation around X-axis)")
    parser.add_argument("--pitch", type=float, help="Pitch angle (rotation around Y-axis)")
    parser.add_argument("--yaw", type=float, help="Yaw angle (rotation around Z-axis)")
    
    # Quaternion inputs
    parser.add_argument("--x", type=float, help="Quaternion x component")
    parser.add_argument("--y", type=float, help="Quaternion y component")
    parser.add_argument("--z", type=float, help="Quaternion z component")
    parser.add_argument("--w", type=float, help="Quaternion w component")
    
    # Visualization options
    parser.add_argument("--plot", action="store_true", help="Show 3D visualization")
    parser.add_argument("--save", type=str, help="Save plot to file (implies --plot)")
    
    args = parser.parse_args()
    
    # If save is specified, enable plotting
    if args.save:
        args.plot = True
    
    # Check for visualization availability
    if args.plot and not VISUALIZATION_AVAILABLE:
        print("Error: Visualization requires matplotlib and numpy.")
        print("Install with: pip install matplotlib numpy --break-system-packages")
        sys.exit(1)
    
    use_degrees = not args.radians
    
    try:
        if args.mode == "e2q":
            # Euler to Quaternion
            if args.roll is None or args.pitch is None or args.yaw is None:
                raise ValueError("--roll, --pitch, and --yaw are required for e2q mode")
            
            euler = EulerAngles(args.roll, args.pitch, args.yaw)
            quat = euler_to_quaternion(euler, degrees=use_degrees)
            
            print_conversion_result("e2q", euler, quat)
            
            if args.plot:
                plot_rotation(euler, quat, save_path=args.save)
            
        elif args.mode == "q2e":
            # Quaternion to Euler
            if args.x is None or args.y is None or args.z is None or args.w is None:
                raise ValueError("--x, --y, --z, and --w are required for q2e mode")
            
            quat = Quaternion(args.x, args.y, args.z, args.w)
            euler = quaternion_to_euler(quat, degrees=use_degrees)
            
            print_conversion_result("q2e", euler, quat)
            
            if args.plot:
                plot_rotation(euler, quat, save_path=args.save)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()