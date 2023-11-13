import React from 'react'
import { useNavigate } from 'react-router-dom';

function Init() {
    const navigate = useNavigate();

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: '300px' }}>
                <div>
                    <button type="button" className="btn btn-primary" style={{ marginRight: '10px', padding: '15px 30px' }} onClick={() => navigate('/login')}>Vino Pharmacy Shift</button>
                    <button type="button" className="btn btn-secondary" style={{ padding: '15px 30px' }} onClick={() => navigate('/gaming')}>Vino Gaming Shift</button>
                </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: '20px' }}>
                <h3>"Unlocking Real-time Visual Insights: Empowering AI/ML Applications with OpenShift and Intel."</h3>

            </div>
        </div>
    )
}

export default Init
