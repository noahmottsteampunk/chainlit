import { Plug } from 'lucide-react';
import { useState } from 'react';
import { useRecoilState } from 'recoil';

import { mcpState, useConfig } from '@chainlit/react-client';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from '@/components/ui/tooltip';

import AnimatedPlugIcon from './AnimatedPlugIcon';
import { McpList } from './List';

interface Props {
  disabled?: boolean;
}

const McpButton = ({ disabled }: Props) => {
  const { config } = useConfig();
  const [mcps] = useRecoilState(mcpState);

  const [open, setOpen] = useState(false);

  const allowSse = !!config?.features.mcp?.sse?.enabled;
  const allowStdio = !!config?.features.mcp?.stdio?.enabled;
  const allowHttp = !!config?.features.mcp?.streamable_http?.enabled;
  const allowMcp = !!config?.features.mcp?.enabled;

  if (!allowMcp || (!allowSse && !allowStdio && !allowHttp)) return null;

  const connectedMcps = mcps.filter((mcp) => mcp.status === 'connected');

  const mcpLoading = mcps.find((mcp) => mcp.status === 'connecting');

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                disabled={disabled}
                variant="ghost"
                size="icon"
                className="hover:bg-muted relative"
              >
                {mcpLoading ? (
                  <AnimatedPlugIcon className="!size-5" />
                ) : (
                  <Plug className="!size-5" />
                )}
                {connectedMcps.length > 0 && (
                  <span className="absolute top-0.5 right-0.5 bg-primary text-primary-foreground text-[8px] font-medium rounded-full w-3 h-3 flex items-center justify-center">
                    {connectedMcps.length}
                  </span>
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>MCP Servers</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </DialogTrigger>
      <DialogContent
        id="mcp-servers"
        className="min-w-[50vw] max-h-[85vh] flex flex-col gap-6 bg-background overflow-y-auto"
      >
        <DialogHeader>
          <DialogTitle>Authorized MCP Servers</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <McpList onAddNewClick={() => {}} />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default McpButton;
