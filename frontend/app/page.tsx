"use client";

import { v4 as uuidv4 } from "uuid";
import { ChatWindow } from "./components/ChatWindow";
import { ToastContainer } from "react-toastify";
import { useEffect } from "react";

import { ChakraProvider } from "@chakra-ui/react";

// Clear any potential Directus-related properties from the window object
if (typeof window !== 'undefined') {
  // @ts-ignore
  if (window.__DIRECTUS_CONFIG) {
    // @ts-ignore
    delete window.__DIRECTUS_CONFIG;
  }
}

export default function Home() {
  // Add debugging code
  useEffect(() => {
    console.log("Home component mounted");
    // Write debugging info to the document as a fallback
    const debugElement = document.createElement('div');
    debugElement.id = 'debug-info';
    debugElement.style.position = 'fixed';
    debugElement.style.top = '10px';
    debugElement.style.left = '10px';
    debugElement.style.padding = '10px';
    debugElement.style.background = 'rgba(255,255,255,0.9)';
    debugElement.style.color = 'black';
    debugElement.style.zIndex = '9999';
    debugElement.innerHTML = 'Debugging: Page loaded at ' + new Date().toISOString();
    document.body.appendChild(debugElement);
  }, []);

  try {
    return (
      <ChakraProvider>
        <ToastContainer />
        <ChatWindow conversationId={uuidv4()}></ChatWindow>
      </ChakraProvider>
    );
  } catch (error) {
    // If an error occurs, render a fallback UI
    console.error("Error rendering Home component:", error);
    return (
      <div style={{color: 'white', padding: '20px'}}>
        <h1>Error loading application</h1>
        <p>An error occurred while loading the application.</p>
        <pre>{error instanceof Error ? error.message : String(error)}</pre>
      </div>
    );
  }
}
