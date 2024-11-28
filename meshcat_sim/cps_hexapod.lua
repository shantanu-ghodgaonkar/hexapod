sim=require'sim'
simIK=require'simIK'

function sysCall_init()
    antBase=sim.getObject('../base')
    legBase=sim.getObject('../legBase')
    
    local simLegTips={}
    simLegTargets={}
    for i=1,6,1 do
        simLegTips[i]=sim.getObject('../footTip'..i-1)
        simLegTargets[i]=sim.getObject('../footTarget'..i-1)
    end
    initialPos={}
    for i=1,6,1 do
        initialPos[i]=sim.getObjectPosition(simLegTips[i],legBase)
    end
    legMovementIndex={1,4,2,6,3,5}
    stepProgression=0
    realMovementStrength=0
    
    --IK:
    local simBase=sim.getObject('..')
    ikEnv=simIK.createEnvironment()
    -- Prepare the ik group, using the convenience function 'simIK.addElementFromScene':
    ikGroup=simIK.createGroup(ikEnv)
    for i=1,#simLegTips,1 do
        simIK.addElementFromScene(ikEnv,ikGroup,simBase,simLegTips[i],simLegTargets[i],simIK.constraint_position)
    end
    
    movData={}
    movData.vel=0.5
    movData.amplitude=0.16
    movData.height=0.04
    movData.dir=0
    movData.rot=0
    movData.strength=0
end

function sysCall_actuation()
    dt=sim.getSimulationTimeStep()
    
    dx=movData.strength-realMovementStrength
    if (math.abs(dx)>dt*0.1) then
        dx=math.abs(dx)*dt*0.5/dx
    end
    realMovementStrength=realMovementStrength+dx
    
    
    for leg=1,6,1 do
        sp=(stepProgression+(legMovementIndex[leg]-1)/6) % 1
        offset={0,0,0}
        if (sp<(1/3)) then
            offset[1]=sp*3*movData.amplitude/2
        else
            if (sp<(1/3+1/6)) then
                s=sp-1/3
                offset[1]=movData.amplitude/2-movData.amplitude*s*6/2
                offset[3]=s*6*movData.height
            else
                if (sp<(2/3)) then
                    s=sp-1/3-1/6
                    offset[1]=-movData.amplitude*s*6/2
                    offset[3]=(1-s*6)*movData.height
                else
                    s=sp-2/3
                    offset[1]=-movData.amplitude*(1-s*3)/2
                end
            end
        end
        md=movData.dir+math.abs(movData.rot)*math.atan2(initialPos[leg][1]*movData.rot,-initialPos[leg][2]*movData.rot)
        offset2={offset[1]*math.cos(md)*realMovementStrength,offset[1]*math.sin(md)*realMovementStrength,offset[3]*realMovementStrength}
        p={initialPos[leg][1]+offset2[1],initialPos[leg][2]+offset2[2],initialPos[leg][3]+offset2[3]}
        sim.setObjectPosition(simLegTargets[leg],p,legBase) -- We simply set the desired foot position. IK is handled after that
    end
    simIK.handleGroup(ikEnv,ikGroup,{syncWorlds=true,allowError=true})
    
    stepProgression=stepProgression+dt*movData.vel
end

setStepMode=function(stepVelocity,stepAmplitude,stepHeight,movementDirection,rotationMode,movementStrength)
    movData={}
    movData.vel=stepVelocity
    movData.amplitude=stepAmplitude
    movData.height=stepHeight
    movData.dir=math.pi*movementDirection/180
    movData.rot=rotationMode
    movData.strength=movementStrength
end

function moveToPose(obj,relObj,pos,euler,vel,accel)
    local params = {
        object = obj,
        relObject = relObj,
        targetPose = sim.buildPose(pos, euler),
        maxVel = {vel},
        maxAccel = {accel},
        maxJerk = {0.1},
        metric = {1, 1, 1, 0.1},
    }
    sim.moveToPose(params)
end

moveBody=function(index)
    local p={initialP[1],initialP[2],initialP[3]}
    local o={initialO[1],initialO[2],initialO[3]}
    moveToPose(legBase,antBase,p,o,vel,accel)
    if (index==0) then
        -- up/down
        p[3]=p[3]-0.03*sizeFactor
        moveToPose(legBase,antBase,p,o,vel*2,accel)
        p[3]=p[3]+0.03*sizeFactor
        moveToPose(legBase,antBase,p,o,vel*2,accel)
    end
    if (index==1) then
        -- 4x twisting
        o[1]=o[1]+5*math.pi/180
        o[2]=o[2]-05*math.pi/180
        o[3]=o[3]-15*math.pi/180
        p[1]=p[1]-0.03*sizeFactor
        p[2]=p[2]+0.015*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[1]=o[1]-10*math.pi/180
        o[3]=o[3]+30*math.pi/180
        p[2]=p[2]-0.04*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[1]=o[1]+10*math.pi/180
        o[2]=o[2]+10*math.pi/180
        p[2]=p[2]+0.03*sizeFactor
        p[1]=p[1]+0.06*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[1]=o[1]-10*math.pi/180
        o[3]=o[3]-30*math.pi/180
        p[2]=p[2]-0.03*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
    end
    if (index==2) then
        -- rolling
        p[3]=p[3]-0.0*sizeFactor
        o[1]=o[1]+17*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[1]=o[1]-34*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[1]=o[1]+17*math.pi/180
        p[3]=p[3]+0.0*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
    end
    if (index==3) then
        -- pitching
        p[3]=p[3]-0.0*sizeFactor
        o[2]=o[2]+15*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[2]=o[2]-30*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[2]=o[2]+15*math.pi/180
        p[3]=p[3]+0.0*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
    end
    if (index==4) then
        -- yawing
        p[3]=p[3]+0.0*sizeFactor
        o[3]=o[3]+30*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[3]=o[3]-60*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        o[3]=o[3]+30*math.pi/180
        moveToPose(legBase,antBase,p,o,vel,accel)
        p[3]=p[3]-0.0*sizeFactor
        moveToPose(legBase,antBase,p,o,vel,accel)
    end
end

function sysCall_thread()
    sizeFactor=sim.getObjectSizeFactor(antBase)
    vel=0.05
    accel=0.05
    initialP={0,0,0}
    initialO={0,0,0}
    initialP[3]=initialP[3]-0.03*sizeFactor
    moveToPose(legBase,antBase,initialP,initialO,vel,accel)

    stepHeight=0.02*sizeFactor
    maxWalkingStepSize=0.11*sizeFactor
    walkingVel=0.9

    -- On the spot movement:
    setStepMode(walkingVel,maxWalkingStepSize,stepHeight,0,0,0)
    moveBody(0)
    moveBody(1)
    moveBody(2)
    moveBody(3)
    moveBody(4)

    -- Forward walk while keeping a fixed body posture:
    setStepMode(walkingVel,maxWalkingStepSize,stepHeight,0,0,1)
    sim.wait(12)
    for i=1,27,1 do
        setStepMode(walkingVel,maxWalkingStepSize,stepHeight,10*i,0,1)
        sim.wait(0.5)
    end
    -- Stop:
    setStepMode(walkingVel,maxWalkingStepSize,stepHeight,270,0,0)
    sim.wait(2)

    -- Forward walk while changing the body posture:
    setStepMode(walkingVel,maxWalkingStepSize*0.5,stepHeight,0,0,1)
    moveBody(0)
    moveBody(1)
    moveBody(2)
    moveBody(3)
    moveBody(4)
    -- Stop:
    setStepMode(walkingVel,maxWalkingStepSize*0.5,stepHeight,0,0,0)
    sim.wait(2)

    -- Rotate on the spot:
    setStepMode(walkingVel,maxWalkingStepSize*0.5,stepHeight,0,1,1)
    sim.wait(24)
    -- Stop:
    setStepMode(walkingVel,maxWalkingStepSize*0.5,stepHeight,0,0,0)
end

