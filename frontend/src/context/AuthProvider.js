import React, { createContext, useState, useEffect } from 'react';

export const AuthContext = createContext({});

export const AuthProvider = ({ children }) => {
    const [auth, setAuth] = useState(() => {
        const storedToken = localStorage.getItem('authToken');
        const storedUser = localStorage.getItem('authUser');
        return {
            token: storedToken || null,
            user: storedUser || null
        };
    });

    // Persist auth state to localStorage whenever it changes
    useEffect(() => {
        if (auth?.token) {
            localStorage.setItem('authToken', auth.token);
            if (auth.user) {
                localStorage.setItem('authUser', auth.user);
            }
            console.log('Auth state updated and persisted:', auth.user);
        } else {
            localStorage.removeItem('authToken');
            localStorage.removeItem('authUser');
            console.log('Auth token cleared');
        }
    }, [auth]);

    const logout = () => {
        console.log('Logging out');
        localStorage.removeItem('authToken');
        localStorage.removeItem('authUser');
        setAuth({ token: null, user: null });
    };

    return (
        <AuthContext.Provider value={{ auth, setAuth, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export default AuthProvider;
