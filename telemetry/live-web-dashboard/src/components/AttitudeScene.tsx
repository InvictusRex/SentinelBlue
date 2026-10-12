import { useEffect, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { CanvasTexture, Euler, Quaternion, Shape, Vector3 } from "three";
import type { AttitudeState } from "../types/telemetry";

function AxisLabel({ text, position, color }: { text: string; position: [number, number, number]; color: string }) {
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 64;
    const context = canvas.getContext("2d")!;
    context.font = "500 38px monospace";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = color;
    context.fillText(text, 64, 32);
    return new CanvasTexture(canvas);
  }, [text, color]);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <sprite position={position} scale={[0.8, 0.4, 1]}>
      <spriteMaterial map={texture} transparent depthTest={false} />
    </sprite>
  );
}

function Arrow({ roll, pitch, yaw }: AttitudeState) {
  const shape = useMemo(() => {
    const outline = new Shape();
    outline.moveTo(0.85, 0);
    outline.lineTo(-0.55, 0.55);
    outline.lineTo(-0.2, 0);
    outline.lineTo(-0.55, -0.55);
    outline.closePath();
    return outline;
  }, []);
  // Body X is forward. Convert aviation right/down axes into a Z-up scene.
  const orientation = useMemo(() => new Quaternion().setFromEuler(new Euler(
    roll * Math.PI / 180, -pitch * Math.PI / 180, -yaw * Math.PI / 180, "ZYX"
  )), [roll, pitch, yaw]);
  return (
    <group quaternion={orientation}>
      <mesh position={[0, 0, -0.07]}>
        <extrudeGeometry args={[shape, { depth: 0.14, bevelEnabled: false }]} />
        <meshStandardMaterial attach="material-0" color="#2dd4bf" roughness={0.7} />
        <meshStandardMaterial attach="material-1" color="#11685e" roughness={0.8} />
      </mesh>
      <mesh position={[0, 0, -0.071]} rotation={[Math.PI, 0, 0]}>
        <shapeGeometry args={[shape]} />
        <meshStandardMaterial color="#64748b" />
      </mesh>
    </group>
  );
}

export function AttitudeScene({ attitude }: { attitude: AttitudeState }) {
  return (
    <div className="h-[128px] min-w-0 flex-1" role="img" aria-label={`3D attitude, X/Y/Z axes. Roll ${attitude.roll.toFixed(1)}, pitch ${attitude.pitch.toFixed(1)}, yaw ${attitude.yaw.toFixed(1)} degrees.`}>
      <Canvas
        frameloop="demand"
        dpr={[1, 2]}
        camera={{ position: [4, -6, 4], up: [0, 0, 1], fov: 32 }}
        gl={{ alpha: true, antialias: true }}
        fallback={<span className="text-2xs text-hud-dim">3D view requires WebGL</span>}
      >
        <ambientLight intensity={1.3} />
        <directionalLight position={[2, -3, 6]} intensity={2} />
        <gridHelper args={[2.8, 8, "#415361", "#29343e"]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -0.85]} />
        <arrowHelper args={[new Vector3(1, 0, 0), new Vector3(-1.4, 0, -0.85), 2.95, "#ff757e", 0.14, 0.07]} />
        <arrowHelper args={[new Vector3(0, 1, 0), new Vector3(0, -1.4, -0.85), 2.95, "#2dd4bf", 0.14, 0.07]} />
        <arrowHelper args={[new Vector3(0, 0, 1), new Vector3(0, 0, -0.85), 2.25, "#78aaff", 0.14, 0.07]} />
        <AxisLabel text="X" position={[1.8, 0, -0.85]} color="#ff757e" />
        <AxisLabel text="Y" position={[0, 1.8, -0.85]} color="#2dd4bf" />
        <AxisLabel text="Z" position={[0, 0, 1.65]} color="#78aaff" />
        <Arrow {...attitude} />
      </Canvas>
    </div>
  );
}
