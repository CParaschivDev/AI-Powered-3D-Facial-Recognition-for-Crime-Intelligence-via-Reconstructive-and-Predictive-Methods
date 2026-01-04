import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

function FaceMesh({ vertices, faces, position = [0, 0, 0], autoRotate = false, rotSpeed = 0.6 }) {
  const meshRef = useRef();

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    if (vertices && vertices.length > 0) {
      const flatVertices = new Float32Array(vertices.flat());
      geom.setAttribute('position', new THREE.BufferAttribute(flatVertices, 3));
    }
    if (faces && faces.length > 0) {
      const flatFaces = new Uint32Array(faces.flat());
      geom.setIndex(new THREE.BufferAttribute(flatFaces, 1));
    }
    try {
      geom.computeVertexNormals();
    } catch (e) {
      // ignore
    }
    return geom;
  }, [vertices, faces]);

  useFrame((state, delta) => {
    if (autoRotate && meshRef.current) {
      meshRef.current.rotation.y += rotSpeed * 0.2 * delta;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} position={position}>
      <meshStandardMaterial
        color="#b8b8b8"
        roughness={0.3}
        metalness={0.05}
        wireframe={false}
        side={THREE.DoubleSide}
        flatShading={false}
      />
    </mesh>
  );
}

function ThreeViewLayout({ vertices, faces }) {
  const [autoRotate, setAutoRotate] = useState(false);
  const [controlsEnabled, setControlsEnabled] = useState(true);

  const handleWheelCapture = (e) => {
    const evt = e.nativeEvent || e;
    const hasModifier = evt.ctrlKey || evt.shiftKey || evt.altKey || evt.metaKey;
    if (!hasModifier) {
      // Let the page scroll — prevent OrbitControls from also handling the wheel
      e.stopPropagation();
      // do not call preventDefault so the browser can scroll the page
    } else {
      // Modifier pressed: prefer zoom behavior, prevent page from scrolling
      try {
        evt.preventDefault();
      } catch (err) {
        // ignore
      }
      // do not stop propagation — allow OrbitControls to receive the event
    }
  };

  return (
    <div style={{ display: 'flex', gap: '10px', height: '100%', position: 'relative' }}>
      <div style={{ position: 'absolute', right: 12, top: 12, zIndex: 50, display: 'flex', gap: '8px' }}>
        <button onClick={() => setAutoRotate(r => !r)} style={{ padding: '6px 10px', borderRadius: 6 }}>
          {autoRotate ? 'Stop Auto' : 'Auto Rotate'}
        </button>
        <button onClick={() => setControlsEnabled(c => !c)} style={{ padding: '6px 10px', borderRadius: 6 }}>
          {controlsEnabled ? 'Lock Controls' : 'Enable Controls'}
        </button>
      </div>

      {/* Front View */}
      <div style={{ flex: 1, position: 'relative', border: '2px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: '10px', left: '10px', color: '#fff', zIndex: 10, fontSize: '14px', fontWeight: 'bold', background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>
          Front View
        </div>
        <Canvas
          camera={{ position: [0, 0, 150], fov: 50 }}
          style={{ background: '#1a1a1a' }}
          gl={{ antialias: true, alpha: false }}
          onWheelCapture={handleWheelCapture}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[0, 50, 100]} intensity={1.2} />
          <directionalLight position={[0, -30, 50]} intensity={0.4} />
          <pointLight position={[0, 0, 100]} intensity={0.3} />
          <FaceMesh vertices={vertices} faces={faces} autoRotate={autoRotate} />
          <OrbitControls 
            enableRotate={controlsEnabled} 
            enableZoom={controlsEnabled} 
            enablePan={false}
            minDistance={80}
            maxDistance={300}
          />
        </Canvas>
      </div>

      {/* Side View (90 degrees) */}
      <div style={{ flex: 1, position: 'relative', border: '2px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: '10px', left: '10px', color: '#fff', zIndex: 10, fontSize: '14px', fontWeight: 'bold', background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>
          Side View
        </div>
        <Canvas
          camera={{ position: [150, 0, 0], fov: 50 }}
          style={{ background: '#1a1a1a' }}
          gl={{ antialias: true, alpha: false }}
          onWheelCapture={handleWheelCapture}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[100, 50, 0]} intensity={1.2} />
          <directionalLight position={[100, -30, 0]} intensity={0.4} />
          <pointLight position={[100, 0, 0]} intensity={0.3} />
          <FaceMesh vertices={vertices} faces={faces} autoRotate={autoRotate} />
          <OrbitControls 
            enableRotate={controlsEnabled} 
            enableZoom={controlsEnabled} 
            enablePan={false}
            minDistance={80}
            maxDistance={300}
          />
        </Canvas>
      </div>

      {/* Three-Quarter View */}
      <div style={{ flex: 1, position: 'relative', border: '2px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: '10px', left: '10px', color: '#fff', zIndex: 10, fontSize: '14px', fontWeight: 'bold', background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>
          3/4 View
        </div>
        <Canvas
          camera={{ position: [100, 20, 100], fov: 50 }}
          style={{ background: '#1a1a1a' }}
          gl={{ antialias: true, alpha: false }}
          onWheelCapture={handleWheelCapture}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[100, 50, 100]} intensity={1.2} />
          <directionalLight position={[-50, -30, 50]} intensity={0.4} />
          <pointLight position={[50, 0, 100]} intensity={0.3} />
          <FaceMesh vertices={vertices} faces={faces} autoRotate={autoRotate} />
          <OrbitControls 
            enableRotate={controlsEnabled} 
            enableZoom={controlsEnabled} 
            enablePan={false}
            minDistance={80}
            maxDistance={300}
          />
        </Canvas>
      </div>
    </div>
  );
}

function FaceViewer({ reconstructionData }) {
  console.log('FaceViewer received reconstructionData:', reconstructionData);

  if (!reconstructionData) {
    return (
      <div className="face-viewer-placeholder">
        <div className="placeholder-content">
          <div className="placeholder-icon">👤</div>
          <p>3D face model will appear here after reconstruction</p>
          <small style={{ color: '#999', marginTop: '0.5rem' }}>
            Upload an image and wait for processing to complete
          </small>
        </div>
      </div>
    );
  }

  const { vertices, faces } = reconstructionData;
  console.log('FaceViewer - vertices:', vertices ? vertices.length : 'null', 'faces:', faces ? faces.length : 'null');

  return (
    <div className="face-viewer-container" style={{ height: '600px' }}>
      <ThreeViewLayout vertices={vertices} faces={faces} />
    </div>
  );
}

export default FaceViewer;