// src/components/tabs/TabNavigation.tsx
import { TabType } from '../../types';

interface TabButtonProps {
  tab: TabType;
  label: string;
  icon: string;
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
}

const TabButton = ({ tab, label, icon, activeTab, setActiveTab }: TabButtonProps) => (
  <button
    onClick={() => setActiveTab(tab)}
    className={`px-4 py-2 font-semibold text-sm rounded-t-lg transition-all ${
      activeTab === tab
        ? 'bg-white border-t-2 border-l border-r border-blue-500 text-blue-700 -mb-px'
        : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-transparent'
    }`}
  >
    {icon} {label}
  </button>
);

interface TabNavigationProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
}

export function TabNavigation({ activeTab, setActiveTab }: TabNavigationProps) {
  return (
    <div className="mt-4 flex gap-1 border-b border-gray-300">
      <TabButton tab="file" label="Archivo" icon="📦" activeTab={activeTab} setActiveTab={setActiveTab} />
      <TabButton tab="model" label="Modelo" icon="🧠" activeTab={activeTab} setActiveTab={setActiveTab} />
      <TabButton tab="params" label="Parámetros" icon="⚖️" activeTab={activeTab} setActiveTab={setActiveTab} />
      <TabButton tab="visualization" label="Visualización" icon="🔗" activeTab={activeTab} setActiveTab={setActiveTab} />
    </div>
  );
}
